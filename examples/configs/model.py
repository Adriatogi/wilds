model_defaults = { #TODO: Need to know these too
    "vit_b_16": {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    "vit_l_16": {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'bert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'distilbert-base-uncased': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'code-gpt-py': {
        'optimizer': 'AdamW',
        'max_grad_norm': 1.0,
        'scheduler': 'linear_schedule_with_warmup',
    },
    'densenet121': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'wideresnet50': {
        'model_kwargs': {
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet18': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet34': {
        'model_kwargs':{
            'pretrained':True,
        },
        'target_resolution': (224, 224),
    },
    'resnet50': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'resnet101': {
        'model_kwargs': {
            'pretrained': True,
        },
        'target_resolution': (224, 224),
    },
    'gin-virtual': {},
    'resnet18_ms': {
        'target_resolution': (224, 224),
    },
    'logistic_regression': {},
    'unet-seq': {
        'optimizer': 'Adam'
    },
    'fasterrcnn': {
        'model_kwargs': {
            'pretrained_model': True,
            'pretrained_backbone': True,
            'min_size' :1024,
            'max_size' :1024
        }
    }
}
