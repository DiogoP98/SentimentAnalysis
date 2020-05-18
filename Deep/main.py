import RNN
import fine_tune
import utils
import train

if __name__ == "__main__":
    selected_model, checkpoints, dataloader_path, model_path, three_class_problem, test_mode = utils.arg_parser()
    utils.setup_seeds(0)

    if selected_model == "LSTM":
        input_dim, train_loader, val_loader, test_loader = train.pre_process()
        embedding_dim = 50
        hidden_dim = 100
        output_dim = 5
        model = RNN.rnn(input_dim, embedding_dim, hidden_dim, output_dim)
        train.train(model, model_path, selected_model, train_loader, val_loader, test_loader)
    else:
        fine_tune.run_finetune(selected_model, checkpoints, dataloader_path, model_path, three_class_problem, test_mode)
