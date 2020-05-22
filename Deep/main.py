import RNN
import fine_tune
import utils
import RNN_train

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    selected_model, checkpoints, saving_path, three_class_problem, test_mode = utils.arg_parser()
    utils.setup_seeds(0)

    if selected_model == "LSTM":
        input_dim, train_loader, val_loader, test_loader, class_problem = RNN_train.pre_process(three_class_problem)
        embedding_dim = 100
        hidden_dim = 100
        output_dim = class_problem
        model = RNN.rnn(input_dim, embedding_dim, hidden_dim, output_dim)
        if not test_mode:
            RNN_train.train(model, saving_path, selected_model, class_problem, train_loader, val_loader)
        RNN_train.test(model, saving_path, selected_model, class_problem, test_loader)
    else:
        fine_tune.run_finetune(selected_model, checkpoints, saving_path, three_class_problem, test_mode)
