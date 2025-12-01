# main.py (PyTorch version of the original driver)

import sys
import numpy as np

import Configrations
import LinearBlkCodes as lbc
import Iterative_BP_CNN as ibd
import ConvNet
import DataIO  # noqa: F401 (used indirectly via ConvNet / ibd)


def main(argv):
    # Top-level configs
    top_config = Configrations.TopConfig()
    top_config.parse_cmd_line(argv)

    train_config = Configrations.TrainingConfig(top_config)
    net_config = Configrations.NetConfig(top_config)

    # LDPC code
    code = lbc.LDPC(
        top_config.N_code,
        top_config.K_code,
        top_config.file_G,
        top_config.file_H,
    )

    if top_config.function == "GenData":
        # Generate training & test data for CNN denoiser with id currently_trained_net_id
        noise_io = DataIO.NoiseIO(
            top_config.N_code,
            False,
            None,
            top_config.cov_1_2_file,
        )
        print("Generating training data...")
        ibd.generate_noise_samples(
            code,
            top_config,
            net_config,
            train_config,
            top_config.BP_iter_nums_gen_data,
            top_config.currently_trained_net_id,
            "Training",
            noise_io,
            top_config.model_id,
        )

        print("Generating test data...")
        ibd.generate_noise_samples(
            code,
            top_config,
            net_config,
            train_config,
            top_config.BP_iter_nums_gen_data,
            top_config.currently_trained_net_id,
            "Test",
            noise_io,
            top_config.model_id,
        )

    elif top_config.function == "Train":
        # Train CNN denoiser with id currently_trained_net_id
        net_id = top_config.currently_trained_net_id
        conv_net = ConvNet.ConvNet(net_config, train_config, net_id)
        conv_net.train_network(top_config.model_id)

    elif top_config.function == "Simulation":
        # Run end-to-end BER simulation
        batch_size = 5000

        if top_config.analyze_res_noise:
            simutimes_for_anal_res_power = int(
                np.ceil(5e6 / float(top_config.K_code * batch_size))
                * batch_size
            )
            ibd.analyze_residual_noise(
                code,
                top_config,
                net_config,
                simutimes_for_anal_res_power,
                batch_size,
            )

        simutimes_range = np.array(
            [
                np.ceil(1e7 / float(top_config.K_code * batch_size))
                * batch_size,
                np.ceil(1e8 / float(top_config.K_code * batch_size))
                * batch_size,
            ],
            np.int32,
        )

        ibd.simulation_colored_noise(
            code,
            top_config,
            net_config,
            simutimes_range,
            target_err_bits_num=1000,
            batch_size=batch_size,
        )

    else:
        print("Unknown function: %s" % top_config.function)
        print("Valid options: GenData, Train, Simulation")


if __name__ == "__main__":
    main(sys.argv)
