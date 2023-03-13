#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#
import numpy as np
import os
import sys
import cls_data_generator
import seldnet_model
import parameters
import torch
from IPython import embed
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
plot.rcParams.update({'font.size': 22})


def main(argv):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    print('\nLoading the best model and predicting results on the testing split')
    print('\tLoading testing dataset:')
    data_gen_test = cls_data_generator.DataGenerator(
        params=params, split=1, shuffle=False, is_eval=True if params['mode']=='eval' else False
    )
    data_in, data_out = data_gen_test.get_data_sizes()
    dump_figures = True

    # CHOOSE THE MODEL WHOSE OUTPUT YOU WANT TO VISUALIZE 
    checkpoint_name = "models/1_1_foa_dev_split6_model.h5"
    model = seldnet_model.SeldModel(data_in, data_out, params)
    model.eval()
    model.load_state_dict(torch.load(checkpoint_name, map_location=torch.device('cpu')))
    model = model.to(device)
    if dump_figures:
        dump_folder = os.path.join('dump_dir', os.path.basename(checkpoint_name).split('.')[0])
        os.makedirs(dump_folder, exist_ok=True)

    with torch.no_grad():
        file_cnt = 0
        for data, target in data_gen_test.generate():
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            output = model(data)

            # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
            max_nb_doas = output.shape[2]//3
            output = output.view(output.shape[0], output.shape[1], 3, max_nb_doas).transpose(-1, -2)
            target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

            # get pair-wise distance matrix between predicted and reference.
            output, target = output.view(-1, output.shape[-2], output.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])

            output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()

            use_activity_detector = False
            if use_activity_detector:
                activity = (torch.sigmoid(activity_out).cpu().detach().numpy() >0.5)
            mel_spec = data[0][0].cpu()
            foa_iv = data[0][-1].cpu()
            target[target > 1] =0

            plot.figure(figsize=(20,10))
            plot.subplot(321), plot.imshow(torch.transpose(mel_spec, -1, -2))
            plot.subplot(322), plot.imshow(torch.transpose(foa_iv, -1, -2))

            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 0], 'r', lw=2)
            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 1], 'g', lw=2)
            plot.subplot(323), plot.plot(target[:params['label_sequence_length'], 0, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 0], 'r', lw=2)
            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 1], 'g', lw=2)
            plot.subplot(324), plot.plot(target[:params['label_sequence_length'], 1, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])
            if use_activity_detector:
                output[:, 0, 0:3] = activity[:, 0][:, np.newaxis]*output[:, 0, 0:3]
                output[:, 1, 0:3] = activity[:, 1][:, np.newaxis]*output[:, 1, 0:3]

            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 0], 'r', lw=2)
            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 1], 'g', lw=2)
            plot.subplot(325), plot.plot(output[:params['label_sequence_length'], 0, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 0], 'r', lw=2)
            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 1], 'g', lw=2)
            plot.subplot(326), plot.plot(output[:params['label_sequence_length'], 1, 2], 'b', lw=2)
            plot.grid()
            plot.ylim([-1.1, 1.1])

            if dump_figures:
                fig_name = '{}'.format(os.path.join(dump_folder, '{}.png'.format(file_cnt)))
                print('saving figure : {}'.format(fig_name))
                plot.savefig(fig_name, dpi=100)
                plot.close()
                file_cnt += 1
            else:
                plot.show()
            if file_cnt>2:
                break


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


