# Configrations.py (PyTorch-compatible version)

import numpy as np


# ---------------------------------------------------------------------- #
# Top-level configuration (unchanged logic)
# ---------------------------------------------------------------------- #

class TopConfig:
    def __init__(self):
        # select functions to be executed: 'GenData', 'Train', 'Simulation'
        self.function = "Train"

        # Code parameters
        self.N_code = 576
        self.K_code = 432
        self.file_G = "./LDPC_matrix/LDPC_gen_mat_%d_%d.txt" % (
            self.N_code,
            self.K_code,
        )
        self.file_H = "./LDPC_matrix/LDPC_chk_mat_%d_%d.txt" % (
            self.N_code,
            self.K_code,
        )

        # Noise information
        self.blk_len = self.N_code
        self.corr_para = 0.3  # correlation parameter
        # self.corr_para = 0.8  # correlation parameter 
        self.corr_para_simu = self.corr_para
        self.cov_1_2_file = "./Noise/cov_1_2_corr_para%.2f.dat" % self.corr_para
        self.cov_1_2_file_simu = self.cov_1_2_file

        # BP decoding
        self.BP_iter_nums_gen_data = np.array([5])
        self.BP_iter_nums_simu = np.array([5, 5])

        # CNN config
        self.currently_trained_net_id = 0
        self.cnn_net_number = 1
        self.layer_num = 4
        self.filter_sizes = np.array([9, 3, 3, 15])
        self.feature_map_nums = np.array([64, 32, 16, 1])
        self.restore_network_from_file = False
        self.model_id = np.array([0], dtype=np.int32)

        # Training
        self.normality_test_enabled = True
        self.normality_lambda = 1.0
        self.SNR_set_gen_data = np.array(
            [0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=np.float32
        )

        # Simulation
        self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=np.float32)
        self.same_model_all_nets = False
        self.analyze_res_noise = True
        self.update_llr_with_epdf = False

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return

        idx = 1
        while idx < len(argv):
            key = argv[idx]

            # Function
            if key == "-Func":
                self.function = argv[idx + 1]
                print("Function is set to %s" % self.function)

            # Noise information
            elif key == "-CorrPara":
                self.corr_para = float(argv[idx + 1])
                self.cov_1_2_file = "./Noise/cov_1_2_corr_para%.2f.dat" % self.corr_para
                print("Corr para is set to %.2f" % self.corr_para)

            # Simulation options
            elif key == "-UpdateLLR_Epdf":
                self.update_llr_with_epdf = argv[idx + 1] == "True"

            elif key == "-EvalSNR":
                self.eval_SNRs = np.fromstring(argv[idx + 1], np.float32, sep=" ")
                print("eval_SNRs is set to %s" % np.array2string(self.eval_SNRs))

            elif key == "-AnalResNoise":
                self.analyze_res_noise = argv[idx + 1] == "True"
                print(
                    "analyze_res_noise is set to %s"
                    % str(self.analyze_res_noise)
                )

            elif key == "-SimuCorrPara":
                self.corr_para_simu = float(argv[idx + 1])
                self.cov_1_2_file_simu = (
                    "./Noise/cov_1_2_corr_para%.2f.dat" % self.corr_para_simu
                )
                print(
                    "Corr para for simulation is set to %.2f"
                    % self.corr_para_simu
                )

            elif key == "-SameModelAllNets":
                self.same_model_all_nets = argv[idx + 1] == "True"
                print(
                    "same_model_all_nets is set to %s"
                    % str(self.same_model_all_nets)
                )

            # BP iter numbers
            elif key == "-BP_IterForGenData":
                self.BP_iter_nums_gen_data = np.fromstring(
                    argv[idx + 1], np.int32, sep=" "
                )
                print(
                    "BP iter for gen data is set to: %s"
                    % np.array2string(self.BP_iter_nums_gen_data)
                )

            elif key == "-BP_IterForSimu":
                self.BP_iter_nums_simu = np.fromstring(
                    argv[idx + 1], np.int32, sep=" "
                )
                print(
                    "BP iter for simulation is set to: %s"
                    % np.array2string(self.BP_iter_nums_simu)
                )

            # CNN config
            elif key == "-NetNumber":
                self.cnn_net_number = int(argv[idx + 1])

            elif key == "-CNN_Layer":
                self.layer_num = int(argv[idx + 1])
                print("CNN layer number is set to %d" % self.layer_num)

            elif key == "-FilterSize":
                self.filter_sizes = np.fromstring(
                    argv[idx + 1], np.int32, sep=" "
                )
                print(
                    "Filter sizes are set to %s"
                    % np.array2string(self.filter_sizes)
                )

            elif key == "-FeatureMap":
                self.feature_map_nums = np.fromstring(
                    argv[idx + 1], np.int32, sep=" "
                )
                print(
                    "Feature map numbers are set to %s"
                    % np.array2string(self.feature_map_nums)
                )

            # Training options
            elif key == "-ModelId":
                self.model_id = np.fromstring(
                    argv[idx + 1], np.int32, sep=" "
                )
                print(
                    "Model id is set to %s" % (np.array2string(self.model_id))
                )

            elif key == "-NormTest":
                self.normality_test_enabled = argv[idx + 1] == "True"
                print(
                    "Normality test: %s"
                    % str(self.normality_test_enabled)
                )

            elif key == "-NormLambda":
                self.normality_lambda = np.float32(argv[idx + 1])
                print(
                    "Normality lambda is set to %f"
                    % self.normality_lambda
                )

            elif key == "-SNR_GenData":
                self.SNR_set_gen_data = np.fromstring(
                    argv[idx + 1], np.float32, sep=" "
                )
                print(
                    "SNR set for generating data is set to %s."
                    % np.array2string(self.SNR_set_gen_data)
                )

            else:
                print("Invalid parameter %s!" % key)
                exit(0)

            idx += 2


# ---------------------------------------------------------------------- #
# Net / Training config (mostly unchanged)
# ---------------------------------------------------------------------- #

class NetConfig:
    def __init__(self, top_config: TopConfig):
        # network parameters
        if top_config.restore_network_from_file:
            self.restore_layers = top_config.layer_num
        else:
            self.restore_layers = 0

        self.save_layers = top_config.layer_num
        self.total_layers = top_config.layer_num

        self.feature_length = top_config.blk_len
        self.label_length = top_config.blk_len

        self.node_num_each_layer = (
            np.ones(top_config.layer_num, dtype=np.int32)
            * self.feature_length
        )

        # conv net parameters
        self.filter_sizes = top_config.filter_sizes
        self.feature_map_nums = top_config.feature_map_nums
        self.layer_num = top_config.layer_num

        # folders
        self.model_folder = "./model/"
        self.residual_noise_property_folder = self.model_folder


class TrainingConfig:
    def __init__(self, top_config: TopConfig):
        self.corr_para = top_config.corr_para
        self.currently_trained_net_id = top_config.currently_trained_net_id

        # uncommented are a good medium between cpu and gpu quick tests

        # how many epochs to train the CNN
        # self.epoch_num = 10000
        # self.epoch_num = 2000
        self.epoch_num = 800


        # training data info
        # self.training_sample_num = 1999200
        # self.training_sample_num = 14000      # instead of ~2e6
        self.training_sample_num = 7000
        # self.training_minibatch_size = 1400
        self.training_minibatch_size = 700
        self.SNR_set_gen_data = top_config.SNR_set_gen_data

        self.training_feature_file = "./TrainingData/EstNoise_before_cnn%d.dat" % (
            self.currently_trained_net_id
        )
        self.training_label_file = "./TrainingData/RealNoise.dat"

        # test data info
        # self.test_sample_num = 105000
        # self.test_sample_num = 7000           # instead of 105000
        self.test_sample_num = 3500
        self.test_minibatch_size = 3500
        self.test_feature_file = "./TestData/EstNoise_before_cnn%d.dat" % (
            self.currently_trained_net_id
        )
        self.test_label_file = "./TestData/RealNoise.dat"

        # normality test
        self.normality_test_enabled = top_config.normality_test_enabled
        self.normality_lambda = top_config.normality_lambda

        # sanity checks
        if self.test_sample_num % self.test_minibatch_size != 0:
            print(
                "Total_test_samples must be a multiple of test_minibatch_size!"
            )
            exit(0)

        if self.training_sample_num % self.training_minibatch_size != 0:
            print(
                "Total_training_samples must be a multiple of training_minibatch_size!"
            )
            exit(0)

        if (
            self.training_minibatch_size % np.size(self.SNR_set_gen_data) != 0
            or self.test_minibatch_size
            % np.size(self.SNR_set_gen_data)
            != 0
        ):
            print(
                "A batch of training or test data should contain equal amount "
                "of data under different CSNRs!"
            )
            exit(0)