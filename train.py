import torch
from torch.utils.data import DataLoader
from torch import nn
from pytorch_transformers import AdamW, WEIGHTS_NAME, WarmupLinearSchedule
import csv
import numpy as np
import os
import logging
from fp16 import FP16_Module, FP16_Optimizer
from parallel import DataParallelModel, DataParallelCriterion
from collections import OrderedDict
from utils import *
from settings import args, TASK_DICT, init_logging, MODEL_CONFIG, MODEL_CLASS, SPECIAL_TOKENS, CONFIG_CLASS
from settings import TOKENIZER, SPECIAL_TOKEN_IDS, FILL_VAL, SAVE_NAME, FINAL_SAVE_NAME, TOKENS_WEIGHT, CONFIG_NAME
from scheduler import AnnealingLR
from regularizers import REG_TYPES, REG_TYPE_KEYS, Weight_Regularized_AdamW, Weight_Regularized_SGD
from torch.nn import CrossEntropyLoss
import random
import copy
logger = logging.getLogger(__name__)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


RESTORE_NAME=''

def train(task_ids, model):
    tasks = [args.tasks[task_id] for task_id in task_ids]

    logger.info("start to train { task: %s, seq train type: %s }" % (tasks, args.seq_train_type))
    model_dir = get_model_dir(tasks)
    make_dir(model_dir)


    train_dataset = [TASK_DICT[t]["train"] for t in tasks]
    train_extra_data = []
    if "lll" in args.seq_train_type and task_ids[0] > 0 and not args.skip_tasks:

        prev_task = args.tasks[task_ids[0]-1]

        with torch.no_grad():
            create_extra_data(tasks[0], prev_task, model, train_extra_data,ignore_exist=True)
    elif "gem" in args.seq_train_type and task_ids[0] > 0: 
        get_real_data(tasks[0], train_extra_data, accum=False, encode=True)
        args.memory_data.append(train_extra_data)
        train_extra_data = []
    logger.info('extra training data size: {}'.format(len(train_extra_data)))

    if not model:
        # which_model_to_load = model_dir if os.path.isfile(os.path.join(model_dir, FINAL_SAVE_NAME)) else args.model_name
        model = MODEL_CLASS.from_pretrained(args.pretrain_path).cuda()
        model.resize_token_embeddings(len(TOKENIZER))
        if not args.fp32:
            model = FP16_Module(model)

    gen_token = get_gen_token(tasks[0])
    TOKENIZER.add_tokens([gen_token])
    TOKENIZER.save_pretrained(model_dir)
    SPECIAL_TOKENS[tasks[0]] = gen_token
    SPECIAL_TOKEN_IDS[tasks[0]] = TOKENIZER.convert_tokens_to_ids(gen_token)
    logger.info('gen token = {} , gen token id = {}'.format(gen_token, SPECIAL_TOKEN_IDS[tasks[0]]))
    MODEL_CONFIG.vocab_size = len(TOKENIZER)
    MODEL_CONFIG.to_json_file(os.path.join(model_dir,CONFIG_NAME))
    global TOKENS_WEIGHT
    if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
        TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, args.tokens_weight*torch.ones([1]).cuda()))

    if args.skip_tasks and len(tasks) == 1 and args.retrain:
        logger.info("*********** skip task: {} ***********".format(tasks[0]))
        if tasks[0] in args.skip_tasks:
            if len(args.skip_tasks) == 1:
                restore_model_dir = get_model_dir(tasks)
                if not os.path.exists(os.path.join(restore_model_dir, FINAL_SAVE_NAME)) and args.tasks.index(tasks[0]) == 0:
                    restore_model_dir = get_model_dir(tasks, True)
                    logger.info(
                        '%s is not found, we load default model from %s' % (
                        os.path.join(get_model_dir(tasks,False), FINAL_SAVE_NAME),
                        restore_model_dir))
                model_path = os.path.join(restore_model_dir, FINAL_SAVE_NAME)
                config_path = os.path.join(restore_model_dir,CONFIG_NAME)
                model_config = CONFIG_CLASS.from_json_file(config_path)
                model = MODEL_CLASS(model_config).cuda()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                if not args.fp32:
                    model = FP16_Module(model)
                if args.seq_train_type in REG_TYPE_KEYS:
                    logger.info("calulating reg_params ...")
                    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
                    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
                    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
                    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)
                    regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [train_dataloader], tasks[0])
                    regularizer.task_start_do()
                    regularizer.task_end_do()
                    torch.save(model.state_dict(), os.path.join(restore_model_dir, FINAL_SAVE_NAME))
                    logger.info("done reg_params!")
            args.skip_tasks.remove(tasks[0])
            # if args.restore_epoch in ['9','finish',9]:

            return model

    model.resize_token_embeddings(len(TOKENIZER))

    if not args.fp32:  # again because resize_token_embeddings makes embedding layer fp32
        model = FP16_Module(model)

    parallel_model = DataParallelModel(WrapModel(model), args.device_ids)


    train_qadata = QADataset(train_dataset, "train", SPECIAL_TOKEN_IDS[tasks[0]], train_extra_data)
    max_train_batch_size = max(len(train_qadata) // args.min_n_steps, args.min_batch_size)
    train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
    if not args.unbound and args.seq_train_type != "multitask":
        #n_train_epochs = TASK_DICT[tasks[0]]["n_train_epochs"]
        n_train_epochs = args.n_train_epochs[tasks[0]]
    else:
        n_train_epochs = args.n_train_epochs['_'.join(tasks)]
    n_train_optimization_steps = len(train_qadata) * n_train_epochs
    logger.info('len of train dataset: {} , max train batch size {} , num of opt steps: {}'.format(
        len(train_qadata), max_train_batch_size, n_train_optimization_steps))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if "gem" in args.seq_train_type:
        model.task_id = task_ids[0]
        if not hasattr(model, "grad_dims"):
            model.grad_dims = []
            for param in model.parameters():
                model.grad_dims.append(param.data.numel())
        if not hasattr(model, "grads"):
            model.grads = torch.zeros(sum(model.grad_dims),len(args.tasks))
            model.grads = model.grads.cuda()

    if args.seq_train_type in REG_TYPE_KEYS:
        optimizer = Weight_Regularized_AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if not args.fp32:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=None, dynamic_loss_scale=True,
                                   dynamic_loss_args={'scale_window': 100, 'min_scale': 1, 'delayed_shift': 2})

    scheduler = AnnealingLR(optimizer, start_lr=args.learning_rate, warmup_iter=int(args.n_warmup_ratio*len(train_qadata)),
            num_iters=int(n_train_optimization_steps), decay_style=args.decay_style)
    train_loss_fct = DataParallelCriterion(CrossEntropyLoss(ignore_index=FILL_VAL, weight=TOKENS_WEIGHT), args.device_ids)

    if args.seq_train_type in REG_TYPE_KEYS:
        copy_train_dataloader = create_dataloader(train_qadata, "train", max_train_batch_size)
        prev_task = args.tasks[task_ids[0]-1]
        regularizer = REG_TYPES[args.seq_train_type](model, parallel_model, [copy_train_dataloader], tasks[0], prev_task)
        regularizer.task_start_do()

    tot_n_steps = 0
    train_once = TrainStep(model, optimizer, scheduler)
    if "gem" in args.seq_train_type and task_ids[0] != 0:
        gem_step = GEMStep(model, parallel_model, train_loss_fct, optimizer)
    model.train()
    begin_epoch = int(args.restore_epoch) if args.restore_inplace else 0
    args.restore_inplace=False
    for ep in range(begin_epoch,n_train_epochs):
        cum_loss, cum_qa_loss, cum_lm_loss, cum_q_lm_loss, cur_n_inputs = 0, 0, 0, 0, 0
        for n_steps, (_, _, cqa, _, Y, gen_X, gen_Y, gen_X_q, gen_Y_q,is_extra) in enumerate(train_dataloader):

            n_inputs = sum(_cqa.shape[0] for _cqa in cqa)

            for i in range(len(cqa)):
                cqa[i] = (cqa[i].to(args.device_ids[i]),)
                Y[i] = Y[i].to(args.device_ids[i])
                gen_X[i] = (gen_X[i].to(args.device_ids[i]),)
                gen_Y[i] = gen_Y[i].to(args.device_ids[i])
                gen_X_q[i] = (gen_X_q[i].to(args.device_ids[i]),)
                gen_Y_q[i] = gen_Y_q[i].to(args.device_ids[i])
                is_extra[i] = is_extra[i].to(args.device_ids[i])

            
            if args.train_question and args.seq_train_type == 'lll':
                losses = get_losses_q(parallel_model, cqa, Y, gen_X, gen_Y, gen_X_q, gen_Y_q, is_extra, train_loss_fct)
            else:
                losses = get_losses(parallel_model, cqa, Y, gen_X, gen_Y, train_loss_fct)

            loss = sum(losses.values())
            if "gem" in args.seq_train_type and task_ids[0] != 0:
                gem_step(task_ids[0])
            train_once(loss, n_inputs)

            qa_loss = losses['qa_loss'].item() * n_inputs
            lm_loss = losses['lm_loss'].item() * n_inputs
            q_lm_loss = losses['q_lm_loss'].item()*n_inputs if args.train_question else 0.0
            cum_loss += (qa_loss + lm_loss +  q_lm_loss)
            cum_qa_loss += qa_loss
            cum_lm_loss += lm_loss
            cum_q_lm_loss += q_lm_loss
            cur_n_inputs += n_inputs

            if (n_steps + 1 ) % args.logging_steps == 0:
                logger.info('progress {:.3f} , lr {:.1E} , loss {:.3f} , qa loss {:.3f} , lm loss {:.3f} , q_lm loss {:.3f}, avg batch size {:.1f}'.format(
                    ep + cur_n_inputs/len(train_qadata), scheduler.get_lr(),
                    cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
                    cum_q_lm_loss/cur_n_inputs,
                    cur_n_inputs/(n_steps + 1)
                ))

        torch.save(model.state_dict(), os.path.join(model_dir, SAVE_NAME+str(ep+1)))
        tot_n_steps += (n_steps + 1)
        logger.info('epoch {}/{} done , tot steps {} , lr {:.1E} , loss {:.2f} , qa loss {:.2f} , lm loss {:.2f} ,  q_lm loss {:.3f} , avg batch size {:.1f}'.format(
            ep+1, n_train_epochs, tot_n_steps, scheduler.get_lr(),
            cum_loss/cur_n_inputs, cum_qa_loss/cur_n_inputs, cum_lm_loss/cur_n_inputs,
            cum_q_lm_loss/cur_n_inputs, cur_n_inputs/(n_steps+1)
        ))

    # task end do for reg
    if args.seq_train_type in REG_TYPE_KEYS:
        regularizer.task_end_do()
    torch.save(model.state_dict(), os.path.join(model_dir, FINAL_SAVE_NAME))

    return model


def restore_model(task_load,epoch,model_path=None):
    logger.info('Restore model: task_load=%s,epoch=%s'%(task_load,epoch))
    device = torch.device('cuda:%s' % args.device_ids[0])
    model_dir = get_model_dir([task_load])
    logger.info('Restore model: model_root_dir=%s'%model_dir)
    if not os.path.exists(os.path.join(model_dir, 'model-{}'.format(epoch))) and args.tasks.index(task_load)==0 :
        model_dir = get_model_dir([task_load], True)
        logger.info(
            '%s is not found, we load default model from %s' % (os.path.join(model_dir, 'model-{}'.format(epoch)),
                                                                model_dir))

    model_path = os.path.join(model_dir, 'model-{}'.format(epoch))
    config_path = os.path.join(model_dir, CONFIG_NAME)



    lastest_task_idx=args.tasks.index(task_load) + 1
    global TOKENS_WEIGHT
    if args.add_task_tokens:
        for task_name in args.tasks[:lastest_task_idx]:
            gen_token = get_gen_token(task_name)
            TOKENIZER.add_tokens([gen_token])
            SPECIAL_TOKENS[task_name] = gen_token
            SPECIAL_TOKEN_IDS[task_name] = TOKENIZER.convert_tokens_to_ids(gen_token)
            if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
                TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, args.tokens_weight*torch.ones([1]).cuda()))
            #print('special_tokens=',SPECIAL_TOKENS,'\nspecial_tokens_ids=',SPECIAL_TOKEN_IDS)
    else:
        gen_token = get_gen_token(task_load)
        TOKENIZER.add_tokens([gen_token])
        SPECIAL_TOKENS[task_load] = gen_token
        SPECIAL_TOKEN_IDS[task_load] = TOKENIZER.convert_tokens_to_ids(gen_token)
        if len(TOKENIZER) != TOKENS_WEIGHT.shape[0]:
            TOKENS_WEIGHT = torch.cat((TOKENS_WEIGHT, args.tokens_weight*torch.ones([1]).cuda()))
    model_config = CONFIG_CLASS.from_json_file(config_path)
    model = MODEL_CLASS(model_config).to(device)#.train()
    if args.tasks[0]=='wikisql' and task_load=='wikisql':
        with open(model_path, 'rb') as f:
            state_dict = pkl.load(f)
            #print(state_dict.keys())
            for k,v in state_dict.items():
                state_dict[k] = v.data
            model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(model_path, map_location='cuda:%s' % args.device_ids[0])
        model.load_state_dict(state_dict)
    #print('special_tokens=',SPECIAL_TOKENS,'\nspecial_tokens_ids=',SPECIAL_TOKEN_IDS)
    
    if not args.fp32:
        model = FP16_Module(model)
    return model,model_path


if __name__ == '__main__':

    if not args.debug:
        logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
        logging.getLogger("pytorch_transformers.tokenization_utils").setLevel(logging.CRITICAL)

    make_dir(args.model_dir_root)
    
    init_logging(os.path.join(args.model_dir_root, 'log_train.txt'))
    logger.info('args = {}'.format(str(args)))
    
    
    
    model = None
    if args.seq_train_type == "multitask":
        model = train(list(range(len(args.tasks))), model)
    else:
        if args.unbound:
            TASK_DICT = lll_unbound_setting(split_size=args.unbound)
        
        begin_id=0
        lr=args.learning_rate      
        
        for task_id in range(begin_id,len(args.tasks)):
            model = train([task_id], model)
            args.learning_rate=lr
