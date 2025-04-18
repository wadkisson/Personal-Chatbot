A personal AI therapist chatbot I made, completly from scratch!

A breif decription for all the files:

-tinyS.txt: Ignore. This was a small toy dataset I used to test my models architecture in the very early stages of the project.

-model.py: This is the model architecture.

-main_script.py: This is the main script for the model's pre-training.

-fineweb_loader.py: This is the script I pulled straight from a tutorial I followed on how to download the Fineweb dataset.

-loss_history.txt: Recordings of the loss at every pre-training step.

-val_loss_history.txt: Recordings of the validation loss at every 50 pre-training steps.

-samples.txt: A sample from the model's pre-training at every step I recorded the validation loss.

-fine_tune.py: This is the main script for the model's fine-tuning.

-finet_val_samples.txt: Recordings of the validation loss at every 50 fine-tuning steps, along with a sample from the model at that step.

-pretrain_loss.png: Graph representation of the pre-training loss over time.

-pretrain_val_loss.png: Graph representation of the pre-training validation loss over time.

-finetune_val_loss.png: Graph representation of the fine-tuning validation loss over time.

---------------------------------------------------------------------------------------------------------------------------------------------
Project explanation:

This was an educational project for me — how could I make something that would take me from zero machine learning knowledge to a postion where I feel confident in my abilities to develop and test models?

I trained this model on 38100 training steps on the FinewebEDU 10BT data set. My loss settled to around 3.1. I then fine-tuned my model on a small dataset that simulated therapy conversations: https://huggingface.co/datasets/jerryjalapeno/nart-100k-synthetic

My fine-tuned model settled to a loss around 1.45, and I'm quite pleased with its performance so far!

This project was designed to be completly educational for myself, so I will be keeping the weights private and the model offline. You can check out its performance in pre-training and fine-tuning in the text and graph files!

Thanks for reading guys. More projects coming soon :)
