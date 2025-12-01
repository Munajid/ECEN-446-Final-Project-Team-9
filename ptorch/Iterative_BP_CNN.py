# Iterative_BP_CNN.py (PyTorch version)

import datetime
import numpy as np
import torch

import BP_Decoder
import ConvNet
import LinearBlkCodes as lbc
import DataIO


# ---------------------------------------------------------------------- #
# Helper for empirical distribution
# ---------------------------------------------------------------------- #

def stat_prob(x, prob):
    qstep = 0.01
    min_v = -10
    x = np.reshape(x, [1, np.size(x)])
    hist, _ = np.histogram(
        x,
        int(np.round(2 * (-min_v) / qstep)),
        [min_v, -min_v],
    )
    if np.size(prob) == 0:
        prob = hist
    else:
        prob = prob + hist
    return prob


# ---------------------------------------------------------------------- #
# Denoising and LLR updates (PyTorch inference)
# ---------------------------------------------------------------------- #

def _run_cnn(conv_model: ConvNet.ConvNet, y_np, device):
    """
    conv_model: trained ConvNet
    y_np: numpy array (batch_size, feature_length)
    returns: numpy array (batch_size, feature_length)
    """
    conv_model.eval()
    with torch.no_grad():
        x = torch.from_numpy(y_np.astype(np.float32)).to(device)
        out = conv_model.forward(x)
        out = out.cpu().numpy()
    return out


def denoising_and_calc_LLR_awgn(
    res_noise_power,
    y_receive,
    output_pre_decoder,
    conv_model,
    device,
):
    """
    AWGN-based LLR update: LLR = (s_mod + residual_noise) * 2 / res_noise_power
    """
    # Estimate noise with CNN denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = _run_cnn(conv_model, noise_before_cnn, device)

    # Calculate the LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
    return LLR


def calc_LLR_epdf(prob, s_mod_plus_res_noise):
    qstep = 0.01
    min_v = -10

    idx0 = ((s_mod_plus_res_noise - 1 - min_v) / qstep).astype(np.int32)
    idx0[idx0 < 0] = 0
    idx0[idx0 > np.size(prob) - 1] = np.size(prob) - 1
    p0 = prob[idx0]

    idx1 = ((s_mod_plus_res_noise + 1 - min_v) / qstep).astype(np.int32)
    idx1[idx1 < 0] = 0
    idx1[idx1 > np.size(prob) - 1] = np.size(prob) - 1
    p1 = prob[idx1]

    LLR = np.log((p0 + 1e-7) / (p1 + 1e-7))
    return LLR


def denoising_and_calc_LLR_epdf(
    prob,
    y_receive,
    output_pre_decoder,
    conv_model,
    device,
):
    # Estimate noise with CNN
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    noise_after_cnn = _run_cnn(conv_model, noise_before_cnn, device)

    # Calculate LLR for the next BP decoding using empirical pdf
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = calc_LLR_epdf(prob, s_mod_plus_res_noise)
    return LLR


# ---------------------------------------------------------------------- #
# Simulation under colored noise (PyTorch CNN, original logic preserved)
# ---------------------------------------------------------------------- #

def simulation_colored_noise(
    linear_code,
    top_config,
    net_config,
    simutimes_range,
    target_err_bits_num,
    batch_size,
):
    """
    Simulate BER for iterative BP-CNN decoder under colored noise.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configs
    SNRset = top_config.eval_SNRs
    bp_iter_num = top_config.BP_iter_nums_simu
    noise_io = DataIO.NoiseIO(
        top_config.N_code,
        False,
        None,
        top_config.cov_1_2_file_simu,
        rng_seed=0,
    )
    denoising_net_num = top_config.cnn_net_number
    model_id = top_config.model_id

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    K, N = np.shape(G_matrix)

    if np.size(bp_iter_num) != denoising_net_num + 1:
        print("Error: the length of bp_iter_num is not correct!")
        exit(0)

    # BP decoder
    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size)

    # Build CNN denoisers
    conv_net = {}
    for net_id in range(denoising_net_num):
        if top_config.same_model_all_nets and net_id > 0:
            conv_net[net_id] = conv_net[0]
        else:
            conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id, device=device)
            conv_net[net_id].restore_network_with_model_id(
                model_id[0 : (net_id + 1)]
            )

    # Simulation times
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = np.array(divmod(max_simutimes, batch_size), np.int32)
    if residual_times != 0:
        max_batches += 1

    # BER output file
    bp_str = np.array2string(bp_iter_num, separator="_", formatter={"int": lambda d: "%d" % d})
    bp_str = bp_str[1 : (len(bp_str) - 1)]
    ber_file = "%sBER(%d_%d)_BP(%s)" % (net_config.model_folder, N, K, bp_str)
    if top_config.corr_para != top_config.corr_para_simu:
        ber_file = "%s_SimuCorrPara%.2f" % (ber_file, top_config.corr_para_simu)
    if top_config.same_model_all_nets:
        ber_file = "%s_SameModelAllNets" % ber_file
    if top_config.update_llr_with_epdf:
        ber_file = "%s_llrepdf" % ber_file
    if denoising_net_num > 0:
        model_id_str = np.array2string(
            model_id, separator="_", formatter={"int": lambda d: "%d" % d}
        )
        model_id_str = model_id_str[1 : (len(model_id_str) - 1)]
        ber_file = "%s_model%s" % (ber_file, model_id_str)
    if np.size(SNRset) == 1:
        ber_file = "%s_%.1fdB" % (ber_file, SNRset[0])
    ber_file = "%s.txt" % ber_file
    fout_ber = open(ber_file, "wt")

    # Run simulation
    start = datetime.datetime.now()

    for SNR in SNRset:
        real_batch_size = batch_size
        bit_errs_iter = np.zeros(denoising_net_num + 1, dtype=np.int32)
        actual_simutimes = 0
        rng = np.random.RandomState(0)
        noise_io.reset_noise_generator()

        for ik in range(0, int(max_batches)):
            print("Batch %d in total %d batches." % (ik, int(max_batches)), end=" ")

            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times

            x_bits, _, s_mod, ch_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, real_batch_size, noise_io, rng
            )

            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10 * np.log10(1 / (noise_power * 2.0))
            print("Practical EbN0: %.2f" % practical_snr)

            for iter_id in range(0, denoising_net_num + 1):
                # BP decoding
                u_BP_decoded = bp_decoder.decode(
                    LLR.astype(np.float32), bp_iter_num[iter_id]
                )

                if iter_id < denoising_net_num:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter_id].get_res_noise_pdf(model_id).get(
                            np.float32(SNR)
                        )
                        LLR = denoising_and_calc_LLR_epdf(
                            prob,
                            y_receive,
                            u_BP_decoded,
                            conv_net[iter_id],
                            device,
                        )
                    else:
                        res_noise_power = conv_net[iter_id].get_res_noise_power(
                            model_id, SNRset
                        ).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(
                            res_noise_power,
                            y_receive,
                            u_BP_decoded,
                            conv_net[iter_id],
                            device,
                        )

                output_x = linear_code.dec_src_bits(u_BP_decoded)
                bit_errs_iter[iter_id] += np.sum(output_x != x_bits)

            actual_simutimes += real_batch_size
            if (
                bit_errs_iter[denoising_net_num] >= target_err_bits_num
                and actual_simutimes >= min_simutimes
            ):
                break

        print("%d bits are simulated!" % (actual_simutimes * K))
        ber_iter = np.zeros(denoising_net_num + 1, dtype=np.float64)
        fout_ber.write(str(SNR) + "\t")
        for iter_id in range(0, denoising_net_num + 1):
            ber_iter[iter_id] = bit_errs_iter[iter_id] / float(K * actual_simutimes)
            fout_ber.write(str(ber_iter[iter_id]) + "\t")
        fout_ber.write("\n")

    fout_ber.close()
    end = datetime.datetime.now()
    print("Time: %ds" % (end - start).seconds)
    print("end")


# ---------------------------------------------------------------------- #
# Data generation and residual noise analysis (PyTorch CNN)
# ---------------------------------------------------------------------- #

def generate_noise_samples(
    linear_code,
    top_config,
    net_config,
    train_config,
    bp_iter_num,
    net_id_data_for,
    generate_data_for,
    noise_io,
    model_id,
):
    """
    Generate training or test data for CNN net 'net_id_data_for'.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix

    if generate_data_for == "Training":
        SNRset_for_generate = train_config.SNR_set_gen_data
        batch_size_each_SNR = int(
            train_config.training_minibatch_size // np.size(SNRset_for_generate)
        )
        total_batches = int(
            train_config.training_sample_num // train_config.training_minibatch_size
        )
    elif generate_data_for == "Test":
        SNRset_for_generate = train_config.SNR_set_gen_data
        batch_size_each_SNR = int(
            train_config.test_minibatch_size // np.size(SNRset_for_generate)
        )
        total_batches = int(
            train_config.test_sample_num // train_config.test_minibatch_size
        )
    else:
        print("Invalid objective of data generation!")
        exit(0)

    if np.size(bp_iter_num) != net_id_data_for + 1:
        print("Error: the length of bp_iter_num is not correct!")
        exit(0)

    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size_each_SNR)

    # Build CNNs for previous nets (0..net_id_data_for-1)
    conv_net = {}
    for net_id in range(net_id_data_for):
        conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id, device=device)
        conv_net[net_id].restore_network_with_model_id(model_id[0 : (net_id + 1)])

    start = datetime.datetime.now()

    if generate_data_for == "Training":
        fout_est_noise = open(train_config.training_feature_file, "wb")
        fout_real_noise = open(train_config.training_label_file, "wb")
    else:
        fout_est_noise = open(train_config.test_feature_file, "wb")
        fout_real_noise = open(train_config.test_label_file, "wb")

    # Generating data
    for ik in range(0, total_batches):
        for SNR in SNRset_for_generate:
            x_bits, _, _, channel_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, batch_size_each_SNR, noise_io
            )

            for iter_id in range(0, net_id_data_for + 1):
                u_BP_decoded = bp_decoder.decode(
                    LLR.astype(np.float32), bp_iter_num[iter_id]
                )

                if iter_id != net_id_data_for:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter_id].get_res_noise_pdf(model_id).get(
                            np.float32(SNR)
                        )
                        LLR = denoising_and_calc_LLR_epdf(
                            prob,
                            y_receive,
                            u_BP_decoded,
                            conv_net[iter_id],
                            device,
                        )
                    else:
                        res_noise_power = conv_net[iter_id].get_res_noise_power(
                            model_id
                        ).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(
                            res_noise_power,
                            y_receive,
                            u_BP_decoded,
                            conv_net[iter_id],
                            device,
                        )

            # For target CNN net, we collect noise_before_cnn and channel_noise
            noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
            noise_before_cnn = noise_before_cnn.astype(np.float32)
            noise_before_cnn.tofile(fout_est_noise)
            channel_noise.astype(np.float32).tofile(fout_real_noise)

    fout_real_noise.close()
    fout_est_noise.close()

    end = datetime.datetime.now()
    print("Time: %ds" % (end - start).seconds)
    print("end")


def analyze_residual_noise(linear_code, top_config, net_config, simutimes, batch_size):
    """
    Compute residual noise power or empirical distribution of residual noise
    after iterative BP-CNN decoding.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_id_tested = top_config.currently_trained_net_id
    model_id = top_config.model_id
    bp_iter_num = top_config.BP_iter_nums_gen_data[0 : (net_id_tested + 1)]
    noise_io = DataIO.NoiseIO(
        top_config.N_code, False, None, top_config.cov_1_2_file
    )
    SNRset = top_config.eval_SNRs

    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    _, N = np.shape(G_matrix)

    max_batches, residual_times = np.array(divmod(simutimes, batch_size), np.int32)
    print("Real simutimes: %d" % simutimes)
    if residual_times != 0:
        max_batches += 1

    if np.size(bp_iter_num) != net_id_tested + 1:
        print("Error: the length of bp_iter_num is not correct!")
        exit(0)

    bp_decoder = BP_Decoder.BP_NetDecoder(H_matrix, batch_size)

    # Build CNNs for 0..net_id_tested
    conv_net = {}
    for net_id in range(net_id_tested + 1):
        conv_net[net_id] = ConvNet.ConvNet(net_config, None, net_id, device=device)
        conv_net[net_id].restore_network_with_model_id(model_id[0 : (net_id + 1)])

    model_id_str = np.array2string(
        model_id, separator="_", formatter={"int": lambda d: "%d" % d}
    )
    model_id_str = model_id_str[1 : (len(model_id_str) - 1)]
    loss_file_name = "%sresidual_noise_property_netid%d_model%s.txt" % (
        net_config.residual_noise_property_folder,
        net_id_tested,
        model_id_str,
    )
    fout_loss = open(loss_file_name, "wt")

    start = datetime.datetime.now()

    for SNR in SNRset:
        noise_io.reset_noise_generator()
        real_batch_size = batch_size
        loss = 0.0
        prob = np.ones(0)

        for ik in range(0, int(max_batches)):
            print("Batch id: %d" % ik)
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times

            x_bits, _, s_mod, channel_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, real_batch_size, noise_io
            )

            for iter_id in range(0, net_id_tested + 1):
                u_BP_decoded = bp_decoder.decode(
                    LLR.astype(np.float32), bp_iter_num[iter_id]
                )

                noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
                noise_after_cnn = _run_cnn(
                    conv_net[iter_id], noise_before_cnn, device
                )
                s_mod_plus_res_noise = y_receive - noise_after_cnn

                if iter_id < net_id_tested:
                    # Update LLR for next BP iteration
                    if top_config.update_llr_with_epdf:
                        prob_tmp = conv_net[iter_id].get_res_noise_pdf(model_id).get(
                            np.float32(SNR)
                        )
                        LLR = calc_LLR_epdf(prob_tmp, s_mod_plus_res_noise)
                    else:
                        res_noise_power = conv_net[iter_id].get_res_noise_power(
                            model_id
                        ).get(np.float32(SNR))
                        LLR = s_mod_plus_res_noise * 2.0 / res_noise_power

                if top_config.update_llr_with_epdf:
                    prob = stat_prob(s_mod_plus_res_noise - s_mod, prob)
                else:
                    loss += np.sum(
                        np.mean(np.square(s_mod_plus_res_noise - s_mod), axis=1)
                    )

        if top_config.update_llr_with_epdf:
            fout_loss.write(str(SNR) + "\t")
            for i in range(np.size(prob)):
                fout_loss.write(str(prob[i]) + "\t")
            fout_loss.write("\n")
        else:
            loss /= float(simutimes)
            fout_loss.write(str(SNR) + "\t" + str(loss) + "\n")

    fout_loss.close()
    end = datetime.datetime.now()
    print("Time: %ds" % (end - start).seconds)
    print("end")