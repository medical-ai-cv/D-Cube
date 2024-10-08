import torch
import torch.nn as nn
import torch.utils.checkpoint
from module.tools import *
import torch.nn.functional as F

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
        #self.label_emb = nn.Embedding(self.num_classes, embed_dim)
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        x = self.patch_embed(x)
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x

class D_Cube(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                use_checkpoint=False, skip=True):
        super().__init__()
       
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
        self.label_emb = nn.Embedding(self.num_classes, embed_dim)
        self.extras=1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])
        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])
        
        ###D-Cube
        self.norm_f = norm_layer(embed_dim)
        self.conv_channel=nn.Conv2d(771,256,kernel_size=1,stride=1,padding=0)
        self.conv_gray = nn.Conv2d(1024,1,kernel_size=1,stride=1,padding=0)
 
        self.class_layer1=nn.Sequential(
            nn.Conv2d(num_classes, 64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.class_layer2=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.class_layer3=nn.Sequential(
            nn.Conv2d(32,4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True))
        self.class_layer4=nn.Sequential(
            nn.Conv2d(num_classes, 64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True))
        self.class_layer5=nn.Sequential(
            nn.Conv2d(64,32,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True))
        self.class_layer6=nn.Sequential(
            nn.Conv2d(32,4,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True))
        self.norm1_D=nn.BatchNorm2d(64)
        self.norm2_D=nn.BatchNorm2d(32)
        self.norm3_D=nn.BatchNorm2d(4)
        self.fc_layer=nn.Sequential(
        nn.Linear(1024,32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(32, self.num_classes))

        self.conv4resnet = nn.Conv2d(1,3, kernel_size=3, stride=1, padding=1)
        self.sub_features = ExtractSubFeatures(num_classes)

        
        self.softmax = nn.Softmax(dim=1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}
    def forward(self, x, timesteps, ori_image, y=None):
 
        image = self.conv4resnet(ori_image)
        x_guide_feature, score = self.sub_features(image)

        x = self.patch_embed(x)

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed
        skips = []
        feature_list = []

        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)
            img_all_norm = self.norm_f(x)
            feature_list.append(img_all_norm)
        
        x = self.mid_block(x)
        img_all_norm = self.norm_f(x)
        feature_list.append(img_all_norm)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())
            img_all_norm = self.norm_f(x)
            feature_list.append(img_all_norm)
        
        # Modify the concatenated feature based on the p-value of the feature.
        x = torch.cat([feature_list[6], feature_list[8], feature_list[9]], dim=1)

        x = x.unsqueeze(2)
        x = self.conv_channel(x)
        x = torch.squeeze(x, dim=2)  
        x = x.permute(0,2,1)
        x = x.view(-1,512,16,16) 
    
        x = torch.cat((x, x_guide_feature),dim=1)  
        x = self.conv_gray(x)

        att =  x * (score)  
        x = x + att
        
        x1_1 = self.class_layer1(x)
        x1_2 = self.class_layer4(x)
        x = x1_1 + x1_2
        x = self.norm1_D(x)
        x2_1 = self.class_layer2(x)
        x2_2 = self.class_layer5(x)
        x = x2_1 + x2_2
        x = self.norm2_D(x)
        x3_1 = self.class_layer3(x)
        x3_2 = self.class_layer6(x)
        x = x3_1 + x3_2
        x = self.norm3_D(x)
        x = x.contiguous().view(x.size(0), -1)
        output = self.fc_layer(x)

        return output
    
class Cont_UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()
   
        self.label_emb = nn.Embedding(self.num_classes, embed_dim)
        
        self.extras = 1
     

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x1,x2, timesteps, y1=None,y2=None):

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)

        B, L, D = x1.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x1 = torch.cat((time_token, x1), dim=1)
        x2 = torch.cat((time_token, x2), dim=1)
       
        if y1 is not None and y2 is not None:
            label_emb1 = self.label_emb(y1)
            label_emb1 = label_emb1.unsqueeze(dim=1)
            label_emb2 = self.label_emb(y2)
            label_emb2 = label_emb2.unsqueeze(dim=1)
            x1 += label_emb1
            x2 += label_emb2
        
        x1 = x1 + self.pos_embed
        x2 = x2 + self.pos_embed
        
        skips_1 = []
        skips_2 = []
        for bi,blk in enumerate(self.in_blocks):
            x1 = blk(x1)
            x2 = blk(x2)
            skips_1.append(x1)
            skips_2.append(x2)
        

        x1 = self.mid_block(x1)
        x2 = self.mid_block(x2)
       
        mid_feature1=x1.contiguous().view(x1.size(0), -1)
        mid_feature2=x2.contiguous().view(x2.size(0), -1)
    
        for bi,blk in enumerate(self.out_blocks):
            x1 = blk(x1, skips_1.pop())
            x2 = blk(x2, skips_2.pop())
        x1 = self.norm(x1)
        x2 = self.norm(x2)

        x1 = self.decoder_pred(x1)
        x2 = self.decoder_pred(x2)

        assert x1.size(1) == self.extras + L
        x1 = x1[:, self.extras:, :]
        x2 = x2[:, self.extras:, :]
        x1 = unpatchify(x1, self.in_chans)
        x2 = unpatchify(x2, self.in_chans)
        x1 = self.final_layer(x1)
        x2 = self.final_layer(x2)
       
        return x1, x2, mid_feature1, mid_feature2


