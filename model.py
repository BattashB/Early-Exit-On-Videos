import torch
from torch import nn
import models.EE_Resnext as ee_resnext3d
import models.EE_Resnext_eval as ee_resnext3d_eval
import models.EE_Resnext_selflearn as ee_resnext3d_selflearn
import models.EE_Resnext_selflearn_reuse as ee_resnext3d_selflearn_reuse
import models.EE_Resnext_eval_reuse as ee_resnext3d_eval_reuse
from models import  resnext, resnet


def generate_model(opt):
    assert opt.model in [
        'resnext', 'ee_resnext', 'ee_resnext_eval','resnet','ee_resnext_th','ee_resnext_selflearn','ee_resnext_selflearn_reuse','ee_resnext3d_eval_reuse'
    ]

    if opt.pretrain_path:    
        pretrain = torch.load(opt.pretrain_path)

        if opt.n_finetune_classes != 400 and opt.finetune:     
            print("I'm entering here, when I have a pretrained EE model, and I want to fine tune on HMDB51 or UCF101")
            del[pretrain['state_dict']['module.fc.weight']]
            del[pretrain['state_dict']['module.fc.bias']]
            if "ee_" in opt.model:
                del[pretrain['state_dict']['module.exit2.fc_exit0.weight']]
                del[pretrain['state_dict']['module.exit2.fc_exit0.bias']]    
                #del[pretrain['state_dict']['module.exit1.fc_exit0.weight']]
                #del[pretrain['state_dict']['module.exit1.fc_exit0.bias']]

    if opt.model == 'resnet':
        assert opt.model_depth in [18, 34, 50, 101]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
        if opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
        if opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
        if opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    
    if opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    elif opt.model == 'ee_resnext':
        print("model file, number of classes:",opt.n_classes)
        from models.EE_Resnext import get_fine_tuning_parameters
        model = ee_resnext3d.resnext101(
                opt,
                num_classes=opt.n_classes,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    elif opt.model == 'ee_resnext_selflearn':
        print("model file, number of classes:",opt.n_classes)
        from models.EE_Resnext_selflearn import get_fine_tuning_parameters
        model = ee_resnext3d_selflearn.resnext101(
                opt,
                num_classes=opt.n_classes,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    elif opt.model == 'ee_resnext3d_eval_reuse':
        from models.EE_Resnext_selflearn_reuse import get_fine_tuning_parameters
        model = ee_resnext3d_eval_reuse.resnext101(
                opt,
                num_classes=opt.n_classes,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    elif opt.model == 'ee_resnext_selflearn_reuse':
        from models.EE_Resnext_selflearn_reuse import get_fine_tuning_parameters
        model = ee_resnext3d_selflearn_reuse.resnext101(
                opt,
                num_classes=opt.n_classes,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)
    elif opt.model == 'ee_resnext_eval':
        print("model file, number of classes:",opt.n_classes)
        from models.EE_Resnext_eval import get_fine_tuning_parameters
        model = ee_resnext3d_eval.resnext101(
                opt,
                num_classes=opt.n_classes,
                frame_size=opt.frame_size,
                frames_sequence=opt.frames_sequence)

 
                
    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))

            #assert opt.arch == pretrain['arch']

            
            if opt.replace_last_fc :
                print("replacing the fc layer")
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()
            
            #print(model.module.classifier.weight)

            model.load_state_dict(pretrain['state_dict'],strict=False)
            #The 1st loading loads entire model with ee trained on kinetics
            #The 2nd loading loads the backbone that trained on resnext101 kinetics+hmdb51
            if opt.earlyexit_thresholds is not None:
                if opt.dataset == "hmdb51":
                    
                    trained_hmdb51 = torch.load("resnext-101-kinetics-hmdb51_split1.pth")
                   # for k,v in trained_hmdb51['state_dict'].items():
                        
                   #     print(k,"-",v.shape)
                   # stop
                    model.load_state_dict(trained_hmdb51['state_dict'],strict=False)
                if opt.dataset == "ucf101":
                    trained_ucf101 = torch.load("resnext-101-kinetics-ucf101_split1.pth")
                    model.load_state_dict(trained_ucf101['state_dict'],strict=False)

            #print(model.module.classifier.weight)
            freeze_backbone = True
            if freeze_backbone:
                for p in model.parameters():
                    p.requires_grad = False
                try:    
                    for p in model.module.exit1.parameters():
                        p.requires_grad = True
                except:
                    pass
                try:
                    for p in model.module.exit2.parameters():                
                        p.requires_grad = True
                except:
                    pass
                
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format( pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'],strict=False)

        if opt.replace_last_fc :
                model.fc = nn.Linear(model.fc.in_features,
                                            opt.n_finetune_classes)

        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
        return model, parameters

    return model, model.parameters()
