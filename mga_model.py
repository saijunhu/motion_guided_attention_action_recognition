from ResNet import ResNet34, ResNet152, ResNet101,ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
DROPOUT = 0.25
class MvNet(nn.Module):
    def __init__(self,num_segments,num_class,base_model='resnet18'):
        super(MvNet, self).__init__()
        self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        setattr(self.base_model, 'conv1',
                nn.Conv2d(2, 64, 
                            kernel_size=(7, 7),
                            stride=(2, 2),
                            padding=(3, 3),
                            bias=False))
        self.num_segments = num_segments
        self.data_bn = nn.BatchNorm2d(2)


    
    def forward(self, x): 
        x = x.view((-1, ) + x.size()[-3:])
        x = self.data_bn(x)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        conv1_feature = x
        x = self.base_model.layer1(x)
        low_level_feature = x
        x = self.base_model.layer2(x)
        layer2_feature = x
        x = self.base_model.layer3(x)
        layer3_feature = x
        x = self.base_model.layer4(x)
        layer4_feature = x
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc
        fv = x
        x = self.base_model.fc(x)
        output = x.view((-1, self.num_segments) + x.size()[1:])
        # score = torch.mean(output, dim=1)
        return conv1_feature, low_level_feature, layer2_feature, layer3_feature, layer4_feature, fv,output

class IframeNet(nn.Module):
    def __init__(self,num_segments,num_class,mvnet,base_model='resnet101'):
        super(IframeNet, self).__init__()
        self.base_model = getattr(torchvision.models, base_model)(pretrained=True)
        feature_dim = getattr(self.base_model, 'fc').in_features
        setattr(self.base_model, 'fc', nn.Linear(feature_dim+512, num_class))
        self.num_segments = num_segments
        self.data_bn = nn.BatchNorm2d(3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mvnet = mvnet
        

        self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)

        self.conv1x1_layer1_channel_wise = nn.Conv2d(64 * 4, 64 * 4, 1, bias=True)
        self.conv1x1_layer1_spatial = nn.Conv2d(64, 1, 1, bias=True)

        self.conv1x1_layer2_channel_wise = nn.Conv2d(128 * 4, 128 * 4, 1, bias=True)
        self.conv1x1_layer2_spatial = nn.Conv2d(128, 1, 1, bias=True)

        self.conv1x1_layer3_channel_wise = nn.Conv2d(256 * 4, 256 * 4, 1, bias=True)
        self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

        self.conv1x1_layer4_channel_wise = nn.Conv2d(512 * 4, 512 * 4, 1, bias=True)
        self.conv1x1_layer4_spatial = nn.Conv2d(512, 1, 1, bias=True)

        self.init_conv1x1()



    def forward(self, inputs):
        img,mv = inputs
        mv_conv1_feature, mv_layer1_feature, mv_layer2_feature, mv_layer3_feature, mv_layer4_feature, fv ,output_mv= self.mvnet(mv)
        x = img.view((-1, ) + img.size()[-3:])
        x = self.data_bn(x)
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        # conv1_feature = x
        # x = self.MGA_tmc(x, mv_conv1_feature, self.conv1x1_conv1_channel_wise, self.conv1x1_conv1_spatial)
        # after_conv1_feature = x

        x = self.base_model.layer1(x)
        # layer1_feature = x
        # x = self.MGA_tmc(x, mv_layer1_feature, self.conv1x1_layer1_channel_wise, self.conv1x1_layer1_spatial)
        # after_layer1_feature = x

        x = self.base_model.layer2(x)
        layer2_feature = x
        x = self.MGA_tmc(x, mv_layer2_feature, self.conv1x1_layer2_channel_wise, self.conv1x1_layer2_spatial)
        after_layer2_feature = x

        x = self.base_model.layer3(x)
        layer3_feature = x
        x = self.MGA_tmc(x, mv_layer3_feature, self.conv1x1_layer3_channel_wise, self.conv1x1_layer3_spatial)
        after_layer3_feature = x

        x = self.base_model.layer4(x)
        layer4_feature = x
        x = self.MGA_tmc(x, mv_layer4_feature, self.conv1x1_layer4_channel_wise, self.conv1x1_layer4_spatial)
        after_layer4_feature = x

        # img_feat_lst = [conv1_feature, layer1_feature, layer2_feature, layer3_feature, layer4_feature]
        # img_feat_attentioned_lst = [after_conv1_feature, after_layer1_feature, after_layer2_feature, after_layer3_feature,
                                    # after_layer4_feature]
        img_feat_lst = []
        img_feat_attentioned_lst = []

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(torch.cat([x,fv],dim=1))
        # print(output_mv.shape)
        output = x.view((-1, self.num_segments) + x.size()[1:]) + output_mv
        output = torch.mean(output, dim=1)
        return output,img_feat_lst,img_feat_attentioned_lst

    def MGA_tmc(self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial):
        # for example
        # self.conv1x1_conv1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        # self.conv1x1_conv1_spatial = nn.Conv2d(64, 1, 1, bias=True)

        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)

        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def init_conv1x1(self):
        for k, v in self.state_dict().items():
            if 'conv1x1' in k:
                if 'weight' in k:
                    nn.init.kaiming_normal_(v)
                elif 'bias' in k:
                    nn.init.constant_(v, 0)
