from timeit import default_timer as timer
import argparse
from datetime import datetime
from copy import deepcopy
import torchvision.transforms as T
import torchvision.transforms.functional as F
import os
from os.path import join
import numpy as np
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
import numpy as np
import torch
#import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.core_utils import seed_everything, get_args_parser_test
from refgaze import RefGaze
from eval_metrics import get_metrics


def run_model(model, args, num_samples, img_data, text_data, context, context_padding_mask):
    with torch.no_grad():
        pred_boxes, nextwords, targets, token_predictions, scanpaths_x, scanpaths_y = model(img_data = img_data, text_data = text_data, context=context.long(), context_padding_mask=context_padding_mask.bool())
    token_states, xs, ys = list(np.argmax(token_predictions.detach().cpu().numpy(), axis=-1)), list(scanpaths_x.squeeze(-1).detach().cpu().numpy()), \
        list(scanpaths_y.squeeze(-1).detach().cpu().numpy())

    batch_packs = [[] for _ in range(num_samples)]
    batch_terminations = [False for _ in range(num_samples)]
    for i in range(num_samples):
        for state, x, y in zip(token_states[i], xs[i], ys[i]):
            if state == 1:
                break
            elif state == 2:
                batch_terminations[i] = True
                break
            elif state == 0:
                batch_packs[i].append([min(args.im_w-1, int(x)),min(args.im_h-1, int(y))])
            else:
                print("EXCEPTION")
    return batch_packs, batch_terminations
        

def generate_scanpaths(model, tokenizer, test_refs, num_samples, args):
    res = []
    for ref in tqdm(test_refs):
        PIL_image = Image.open(join(args.img_dir, ref['IMAGEFILE'])).convert('RGB')
        tensor_image = T.functional.to_tensor(PIL_image)
        h, w = tensor_image.size()[1:]
        tensor_image = F.normalize(tensor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).repeat(num_samples, 1, 1, 1).cuda()
        
        out_mask = torch.zeros((h, w)).int().unsqueeze(0).repeat(num_samples, 1, 1).cuda()
        text_tokens = ref['REF_WORDS']+ [tokenizer.eos_token] + [tokenizer.pad_token]
        prefix = ''
        context =torch.zeros((num_samples, args.max_context_len, 4)).cuda()
        context_mask = torch.cat([torch.ones(num_samples, args.max_context_len), torch.zeros(num_samples, args.max_pack_len)], dim=-1).cuda()
        scanpath_terminated = [-1 for _ in range(num_samples)]
        scanpaths_X = [[[] for k in range(len(text_tokens))]for j in range(num_samples)]
        scanpaths_Y = deepcopy(scanpaths_X)
        marker=[0 for _ in range(num_samples)]

        for idx, token in enumerate(text_tokens):
            inputs = tokenizer(prefix.strip(), return_tensors="pt", padding='max_length', max_length=args.max_lm_len)
            batch_packs, batch_terminations = run_model(model=model, args=args, num_samples=num_samples, img_data=[tensor_image.cuda(), out_mask.cuda()], \
                text_data=[inputs['input_ids'].repeat(num_samples, 1).cuda(), inputs['attention_mask'].repeat(num_samples,1).cuda()],\
                    context=context, context_padding_mask=context_mask)
            prefix = prefix + token + ' '

            for i in range(len(batch_packs)):
                if scanpath_terminated[i] == -1 and batch_terminations[i] == True:
                    #terminate NOW and mark the time-step
                    scanpath_terminated[i] = int(idx)
                elif scanpath_terminated[i] > -1:
                    #already terminated before
                    pass
                else:
                    pack = batch_packs[i]
                    for order, fix in enumerate(pack):
                        if marker[i] >= args.max_context_len:
                            new_context =torch.zeros((num_samples, args.max_context_len, 4)).cuda()
                            new_context[i, 0:marker[i]-1, :] = context[i, 1:, :].clone()
                            new_context[i, marker[i]-1, :] = torch.tensor([fix[0], fix[1], idx, order]).to(context.device)
                            del context
                            context = deepcopy(new_context.clone())
                            del new_context
                        else:
                            context[i,marker[i], :] = torch.tensor([fix[0], fix[1], idx, order]).to(context.device)
                            context_mask[i, marker[i]] = 0
                            marker[i] += 1
                        scanpaths_X[i][idx].append(int(fix[0]))
                        scanpaths_Y[i][idx].append(int(fix[1]))
                        


        for i in range(num_samples):
            res.append({'REF_ID': ref['REF_ID'], 'X':[scanpaths_X[i][j] for j in range(scanpath_terminated[i])] if scanpath_terminated[i] > -1 else scanpaths_X[i], \
                'Y':[scanpaths_Y[i][j] for j in range(scanpath_terminated[i])] if scanpath_terminated[i] > -1 else scanpaths_Y[i], 'TERMINATIONS':scanpath_terminated[i], 'REPEAT_ID':i})
    return res


def main(args):
    seed_everything(42)
    test_refgazes = json.load(open(join(args.dataset_dir, args.test_file), mode='r'))
    test_refs = []
    test_ref_set = set()
    for case in test_refgazes:
        if case['REF_ID'] not in test_ref_set:
            test_ref_set.add(case['REF_ID'])
            test_refs.append({'REF_ID':case['REF_ID'], 'IMAGEFILE':case['IMAGEFILE'], 'REF_WORDS':case['REF_WORDS']})
    
    

    tokenizer = AutoTokenizer.from_pretrained(args.lm)
    model = RefGaze(args=args).cuda()
    model.eval()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    

    results = generate_scanpaths(model=model, tokenizer=tokenizer, test_refs=test_refs, num_samples=args.num_samples, args=args)

    if not os.path.exists(os.path.abspath('results/')):
        os.mkdir('results/')

    with open('results/'+args.checkpoint.split('/')[-1].replace('pkg','json'), mode='w') as f:
        json.dump(results, f, indent = 6)
        f.close()
        
    get_metrics(predictions=results)


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Refer Test', parents=[get_args_parser_test()])
    args = parser.parse_args()
    main(args)

    

    
