def create_model(opt):
    model = opt['model']

    if model == 'fusion_model':
        from .models import FusionModel as M
    elif model == 'sr_model':
        from .models import SRModel as M
    elif model == 'dn_model':
        from .models import DenoisingModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m