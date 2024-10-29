# FEDFMRL

# FedRCIL

The source code of the paper **"Ensuring Trustworthiness in Decentralized Systems through Federated Distillation and Feature Mixing"** that will be presented in the special session "Special Session on Federated Learning on Big Data" on the 2024 IEEE international conference on Big Data.

# **Installation**

Run pip install -r equirements.txt to install the required packages

# **Proposed method**

![model architecture image](https://github.com/chatzikon/FedRCIL/blob/main/images/FL_IL_scheme_iccv_generic_figure.png)


# **Train**

Use federated_train.py for training a new model. The default parameters are about an experiment with IID data, mu=0.5, buffer_size=4
The majority of the functions required for training and testing are at the script traintest.py. 

The arguments of the main function are explained at the file arguments.txt



# **License**

Our code is released under MIT License (see LICENSE file for details)
