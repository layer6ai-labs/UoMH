# Modified code from https://github.com/kuangliu/pytorch-cifar
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import numpy as np
import torch

def deterministic_shuffle(x):
    is_tensor = False
    if type(x) == torch.Tensor:
        is_tensor = True
        x = np.array(x)

    current_state = np.random.get_state()
    np.random.seed(0)
    np.random.shuffle(x)
    np.random.set_state(current_state)

    if is_tensor:
        x = torch.tensor(x)
        
    return x


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

# Intrinsic dimension of each class divided by sum of intrinsic dimensions
class_weights = [0.007338557403983429,
 0.008470198269290288,
 0.010849306203576636,
 0.010330600312899955,
 0.009123745706762795,
 0.010996726728620915,
 0.010763001530902575,
 0.010618158899558234,
 0.00969991723040435,
 0.007901142620748843,
 0.00872801626958191,
 0.01034228481673111,
 0.011395233821182377,
 0.013068441263614257,
 0.012760941674953065,
 0.010775963429632827,
 0.00900486657389924,
 0.011476940051096361,
 0.011058920447429975,
 0.011888150070605997,
 0.008706343687941546,
 0.010551921793962924,
 0.007745428613977445,
 0.007747466055463483,
 0.00642456234635464,
 0.00900599314269309,
 0.009644800808967518,
 0.010733198566477337,
 0.007064626820126408,
 0.008774380465208786,
 0.010028882138184635,
 0.011300911262660661,
 0.009042812850649996,
 0.011739069748736881,
 0.010803345440879766,
 0.011886472850248587,
 0.009455304732511034,
 0.012755416018656122,
 0.01103252658826491,
 0.007931824847869055,
 0.007634389982826504,
 0.008391342889102508,
 0.011445889115882912,
 0.011923734564018181,
 0.010105549694537247,
 0.009484301054212668,
 0.011171442313288317,
 0.010048801170948543,
 0.012155916766864946,
 0.009462256062653513,
 0.009965878983716982,
 0.012261052359878576,
 0.01114820874560356,
 0.007265486629626704,
 0.012013872798541648,
 0.009812847222230758,
 0.011102091931849684,
 0.00810390885666194,
 0.011884595671742078,
 0.009745120453589133,
 0.006989943469279138,
 0.00712046525162886,
 0.011341670324001182,
 0.01081679460717868,
 0.010676785176092407,
 0.009566135099441993,
 0.012417139862965125,
 0.007767161271572051,
 0.00863276998642797,
 0.007644521696356096,
 0.010483911312238622,
 0.007263571714259775,
 0.010235449816660832,
 0.007987414058204552,
 0.009848912986298013,
 0.008767283617303912,
 0.008616339068702444,
 0.010436386296149172,
 0.012495585389664676,
 0.008696935392684277,
 0.011952118885574662,
 0.012924018231771528,
 0.010171533872680893,
 0.009879828114636108,
 0.009841750534955672,
 0.011088275465020083,
 0.00850891093806028,
 0.010236966766757094,
 0.012531589666021296,
 0.011952547304260666,
 0.011581776851307813,
 0.008804396128839789,
 0.011817937828063965,
 0.009616936954069987,
 0.008826656921328054,
 0.008981945564743918,
 0.010142677900893992,
 0.010491000976172334,
 0.011174224196758141,
 0.007576641127887851]