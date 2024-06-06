import torch
import torch.nn as nn
import os
import traceback
import torchvision

from models.layers import Identity
from utils import load

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    """
    # If load_featurizer_only is True,
    # then split into (featurizer, classifier) for the purposes of loading only the featurizer,
    # before recombining them at the end
    featurize = is_featurizer or config.load_featurizer_only

    if config.model in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'wideresnet50', 'densenet121'):
        if featurize:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)
    elif config.model in ("vit_b_16", "vit_l_16"):
        if featurize:
            featurizer = initialize_torchvision_vit(
                name=config.model, d_out=None, **config.model_kwargs
            )
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_vit(
                name=config.model, d_out=d_out, **config.model_kwargs
            )
    elif 'bert' in config.model:
        if featurize:
            featurizer = initialize_bert_based_model(config, d_out, featurize)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_bert_based_model(config, d_out)

    elif config.model == 'resnet18_ms':  # multispectral resnet 18
        from models.resnet_multispectral import ResNet18
        if featurize:
            featurizer = ResNet18(num_classes=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = ResNet18(num_classes=d_out, **config.model_kwargs)

    elif config.model == 'gin-virtual':
        from models.gnn import GINVirtual
        if featurize:
            featurizer = GINVirtual(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = GINVirtual(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'code-gpt-py':
        from models.code_gpt import GPT2LMHeadLogit, GPT2FeaturizerLMHeadLogit
        from transformers import GPT2Tokenizer
        name = 'microsoft/CodeGPT-small-py'
        tokenizer = GPT2Tokenizer.from_pretrained(name)
        if featurize:
            model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))
            featurizer = model.transformer
            classifier = model.lm_head
            model = (featurizer, classifier)
        else:
            model = GPT2LMHeadLogit.from_pretrained(name)
            model.resize_token_embeddings(len(tokenizer))

    elif config.model == 'logistic_regression':
        assert not featurize, "Featurizer not supported for logistic regression"
        model = nn.Linear(out_features=d_out, **config.model_kwargs)
    elif config.model == 'unet-seq':
        from models.CNN_genome import UNet
        if featurize:
            featurizer = UNet(num_tasks=None, **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = UNet(num_tasks=d_out, **config.model_kwargs)

    elif config.model == 'fasterrcnn':
        if featurize:
            raise NotImplementedError('Featurizer not implemented for detection yet')
        else:
            model = initialize_fasterrcnn_model(config, d_out)
        model.needs_y = True

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # Load pretrained weights from disk using our utils.load function
    if config.pretrained_model_path is not None:
        if config.model in ('code-gpt-py', 'logistic_regression', 'unet-seq'):
            # This has only been tested on some models (mostly vision), so run this code iff we're sure it works
            raise NotImplementedError(f"Model loading not yet tested for {config.model}.")

        if 'bert' not in config.model:  # We've already loaded pretrained weights for bert-based models using the transformers library
            try:
                if featurize:
                    if config.load_featurizer_only:
                        model_to_load = model[0]
                    else:
                        model_to_load = nn.Sequential(*model)
                else:
                    model_to_load = model

                prev_epoch, best_val_metric = load(
                    model_to_load,
                    config.pretrained_model_path,
                    device=config.device)

                print(
                    (f'Initialized model with pretrained weights from {config.pretrained_model_path} ')
                    + (f'previously trained for {prev_epoch} epochs ' if prev_epoch else '')
                    + (f'with previous val metric {best_val_metric} ' if best_val_metric else '')
                )
            except Exception as e:
                print('Something went wrong loading the pretrained model:')
                traceback.print_exc()
                raise

    # Recombine model if we originally split it up just for loading
    if featurize and not is_featurizer:
        model = nn.Sequential(*model)

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if is_featurizer:
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model

class MetaEmbedding(nn.Module):
    def __init__(self, featurizer, classifer, meta_in, meta_out):
        super(MetaEmbedding, self).__init__()
        self.featurizer = featurizer
        self.classifier = classifer
        self.meta_embedding = nn.Linear(meta_in, meta_out)

        d_features = classifer.in_features
        d_out = classifer.out_features
        self.classifier = nn.Linear(d_features + meta_out, d_out)

    def forward(self, x, metadata):
        metadata = metadata.float()

        x = self.featurizer(x)

        # pass through metadata linear and concatenate here
        meta_embedded = self.meta_embedding(metadata)
        x = torch.cat((x, meta_embedded), dim=-1)
        
        # classifier
        x = self.classifier(x)

        return x

def initialize_meta_model(config, d_out):
    print("initializing meta model")
    featurizer, classifier = initialize_model(config, d_out, is_featurizer=True)

    meta_in = config.usable_metadata_len
    if meta_in is None:
        raise ValueError(f"Meta model needs usable_metadata_len config")
    
    # TODO: Dont hard code out
    meta_out = 8
    model = MetaEmbedding(featurizer, classifier, meta_in, meta_out)
    return model


def initialize_torchvision_vit(name, d_out, **kwargs):

    if name == "vit_b_16":
        vit = torchvision.models.vit_b_16(weights="DEFAULT") 
    elif name == 'vit_l_16':
        vit = torchvision.models.vit_l_16(weights="DEFAULT") 
    else:
        raise ValueError(f"Torchvision model {name} not recognized")

    d_features = vit.heads.head.in_features

    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        vit.d_out = d_features
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        vit.d_out = d_out

    vit.heads.head = last_layer

    if 'num_channels' in kwargs and kwargs['num_channels'] > 3:
        new_conv = nn.Conv2d(
                 in_channels=kwargs['num_channels'], out_channels=768, kernel_size=(16, 16), stride=(16, 16)
             )
        
        # copy weights over
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = vit.conv_proj.weight
            new_conv.weight[:, 3:, :, :].zero_()
            new_conv.bias = vit.conv_proj.bias
        
        vit.conv_proj = new_conv

    # embedding (1), encoder (2-13), and last layer norm as a layer (14)
    num_freeze = kwargs.get('num_freeze', 0)
    if num_freeze > 0:
        for p in vit.conv_proj.parameters():
            p.requires_grad = False
        vit.class_token.requires_grad = False

        if num_freeze > 1:
            encoder_layers = vit.encoder.layers
            print(len(encoder_layers))

            for layer in encoder_layers[:num_freeze]:
                for param in layer.parameters():
                    param.requires_grad = False

            vit.encoder.pos_embedding.requires_grad = False
            print("froze position")

            if num_freeze == len(encoder_layers) + 2:
                print("froze ln")
                for p in vit.encoder.ln.parameters():
                    p.requires_grad = False
             
    for layer_name, p in vit.named_parameters():
        print('Layer Name: {}, Frozen: {}'.format(layer_name, not p.requires_grad))
        print()

    def modify_vit_encoder_dropout(vit_model, dropout_rate=0):
        for layer in vit_model.encoder.layers:
            layer.dropout.p = dropout_rate
            for module in layer.mlp:
                if isinstance(module, nn.Dropout):
                    module.p = dropout_rate

        return vit_model
    
    dropout_rate = kwargs.get('dropout', None)
    if dropout_rate:
        vit = modify_vit_encoder_dropout(vit, dropout_rate)
    #print(vit)

    return vit

def initialize_bert_based_model(config, d_out, featurize=False):
    from models.bert.bert import BertClassifier, BertFeaturizer
    from models.bert.distilbert import DistilBertClassifier, DistilBertFeaturizer

    if config.pretrained_model_path:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        config.model_kwargs['state_dict'] = torch.load(config.pretrained_model_path, map_location=config.device)

    if config.model == 'bert-base-uncased':
        if featurize:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    elif config.model == 'distilbert-base-uncased':
        if featurize:
            model = DistilBertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        else:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)
    else:
        raise ValueError(f'Model: {config.model} not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):

    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet18', 'resnet34', 'resnet50', 'resnet101'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)

    return model

def initialize_fasterrcnn_model(config, d_out):
    from models.detection.fasterrcnn import fasterrcnn_resnet50_fpn

    # load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(
        pretrained=config.model_kwargs["pretrained_model"],
        pretrained_backbone=config.model_kwargs["pretrained_backbone"],
        num_classes=d_out,
        min_size=config.model_kwargs["min_size"],
        max_size=config.model_kwargs["max_size"]
        )

    return model
