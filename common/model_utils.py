#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""
# try to use legecy optimizer if possible
try:
    from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD
except:
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.experimental import CosineDecay


def get_pruning_model(model, begin_step, end_step):
    import tensorflow as tf
    from tensorflow_model_optimization.sparsity import keras as sparsity
    if tf.__version__.startswith('2'):
        # model pruning API is not supported in TF 2.0 yet
        raise Exception('model pruning is not fully supported in TF 2.x, Please switch env to TF 1.x for this feature')

    pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                   final_sparsity=0.7,
                                                   begin_step=begin_step,
                                                   end_step=end_step,
                                                   frequency=100)
    }

    pruning_model = sparsity.prune_low_magnitude(model, **pruning_params)
    return pruning_model


# some global value for lr scheduler
# need to update to CLI option in main()
#lr_base = 1e-3
#total_epochs = 250

#def learning_rate_scheduler(epoch, curr_lr, mode='cosine_decay'):
    #lr_power = 0.9
    #lr = curr_lr

    ## adam default lr
    #if mode is 'adam':
        #lr = 0.001

    ## original lr scheduler
    #if mode is 'power_decay':
        #lr = lr_base * ((1 - float(epoch) / total_epochs) ** lr_power)

    ## exponential decay policy
    #if mode is 'exp_decay':
        #lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)

    ## cosine decay policy, including warmup and hold stage
    #if mode is 'cosine_decay':
        ##warmup & hold hyperparams, adjust for your training
        #warmup_epochs = 0
        #hold_base_rate_epochs = 0
        #warmup_lr = lr_base * 0.01
        #lr = 0.5 * lr_base * (1 + np.cos(
             #np.pi * float(epoch - warmup_epochs - hold_base_rate_epochs) /
             #float(total_epochs - warmup_epochs - hold_base_rate_epochs)))

        #if hold_base_rate_epochs > 0 and epoch < warmup_epochs + hold_base_rate_epochs:
            #lr = lr_base

        #if warmup_epochs > 0 and epoch < warmup_epochs:
            #if lr_base < warmup_lr:
                #raise ValueError('learning_rate_base must be larger or equal to '
                                 #'warmup_learning_rate.')
            #slope = (lr_base - warmup_lr) / float(warmup_epochs)
            #warmup_rate = slope * float(epoch) + warmup_lr
            #lr = warmup_rate

    #if mode is 'progressive_drops':
        ## drops as progression proceeds, good for sgd
        #if epoch > 0.9 * total_epochs:
            #lr = 0.0001
        #elif epoch > 0.75 * total_epochs:
            #lr = 0.001
        #elif epoch > 0.5 * total_epochs:
            #lr = 0.01
        #else:
            #lr = 0.1

    #print('learning_rate change to: {}'.format(lr))
    #return lr


def get_lr_scheduler(learning_rate, decay_type, decay_steps):
    if decay_type:
        decay_type = decay_type.lower()

    if decay_type == None:
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, alpha=0.2) # use 0.2*learning_rate as final minimum learning rate
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, end_learning_rate=learning_rate/100)
    elif decay_type == 'piecewise_constant':
        #apply a piecewise constant lr scheduler, including warmup stage
        boundaries = [500, int(decay_steps*0.9), decay_steps]
        values = [0.001, learning_rate, learning_rate/10., learning_rate/100.]
        lr_scheduler = PiecewiseConstantDecay(boundaries=boundaries, values=values)
    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, average_type=None, decay_type='cosine', decay_steps=100000):
    optim_type = optim_type.lower()

    lr_scheduler = get_lr_scheduler(learning_rate, decay_type, decay_steps)

    # NOTE: you can try to use clipnorm/clipvalue to avoid run into nan, especially on new TF versions
    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler, epsilon=1e-7, amsgrad=False, clipnorm=None, clipvalue=None)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler, rho=0.9, momentum=0.0, centered=False, clipnorm=None, clipvalue=None)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler, momentum=0.9, nesterov=False, clipnorm=None, clipvalue=None)
    else:
        raise ValueError('Unsupported optimizer type')

    if average_type:
        optimizer = get_averaged_optimizer(average_type, optimizer)

    return optimizer


def get_averaged_optimizer(average_type, optimizer):
    """
    Apply weights average mechanism in optimizer. Need tensorflow-addons
    which request TF 2.x and have following compatibility table:
    -------------------------------------------------------------
    |    Tensorflow Addons     | Tensorflow |    Python          |
    -------------------------------------------------------------
    | tfa-nightly              | 2.3, 2.4   | 3.6, 3.7, 3.8      |
    -------------------------------------------------------------
    | tensorflow-addons-0.12.0 | 2.3, 2.4   | 3.6, 3.7, 3.8      |
    -------------------------------------------------------------
    | tensorflow-addons-0.11.2 | 2.2, 2.3   | 3.5, 3.6, 3.7, 3.8 |
    -------------------------------------------------------------
    | tensorflow-addons-0.10.0 | 2.2        | 3.5, 3.6, 3.7, 3.8 |
    -------------------------------------------------------------
    | tensorflow-addons-0.9.1  | 2.1, 2.2   | 3.5, 3.6, 3.7      |
    -------------------------------------------------------------
    | tensorflow-addons-0.8.3  | 2.1        | 3.5, 3.6, 3.7      |
    -------------------------------------------------------------
    | tensorflow-addons-0.7.1  | 2.1        | 2.7, 3.5, 3.6, 3.7 |
    -------------------------------------------------------------
    | tensorflow-addons-0.6.0  | 2.0        | 2.7, 3.5, 3.6, 3.7 |
    -------------------------------------------------------------
    """
    import tensorflow_addons as tfa

    average_type = average_type.lower()

    if average_type == None:
        averaged_optimizer = optimizer
    elif average_type == 'ema':
        averaged_optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=0.99)
    elif average_type == 'swa':
        averaged_optimizer = tfa.optimizers.SWA(optimizer, start_averaging=0, average_period=10)
    elif average_type == 'lookahead':
        averaged_optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=6, slow_step_size=0.5)
    else:
        raise ValueError('Unsupported average type')

    return averaged_optimizer

