import torch
import torch.nn as nn
from built_graph import DeepAE
from built_graph import AttentionLayer
from built_graph import built_cos_knn
from built_graph import KNN_Att
from built_graph import Multi_head_attention


class IDMMC(nn.Module):
    def __init__(self, args, config, cptpath):
        super(IDMMC, self).__init__()
        self.args = args
        self.config = config

        assert (
            config["img_hiddens"][-1] == config["txt_hiddens"][-1]
        ), "Inconsistent latent dim!"

        self.latent_dim = config["img_hiddens"][-1]

        dicts = torch.load(cptpath)

        self.imgAE = DeepAE(
            input_dim=config["img_input_dim"],
            hiddens=config["img_hiddens"],
            batchnorm=config["batchnorm"],
        )
        self.imgAE.load_state_dict(dicts["G1_state_dict"])

        self.txtAE = DeepAE(
            input_dim=config["txt_input_dim"],
            hiddens=config["txt_hiddens"],
            batchnorm=config["batchnorm"],
        )
        self.txtAE.load_state_dict(dicts["G2_state_dict"])

        self.img2txt = DeepAE(
            input_dim=self.latent_dim,
            hiddens=config["img2txt_hiddens"],
            batchnorm=config["batchnorm"],
        )
        self.img2txt.load_state_dict(dicts["G12_state_dict"])

        self.txt2img = DeepAE(
            input_dim=self.latent_dim,
            hiddens=config["txt2img_hiddens"],
            batchnorm=config["batchnorm"],
        )
        self.txt2img.load_state_dict(dicts["G21_state_dict"])

        self.img_att = Multi_head_attention(self.latent_dim, self.latent_dim)
        self.txt_att = Multi_head_attention(self.latent_dim, self.latent_dim)

        self.img_att1 = KNN_Att(self.latent_dim, self.latent_dim)
        self.txt_att1 = KNN_Att(self.latent_dim, self.latent_dim)

        self.Cross_att1 = AttentionLayer(self.latent_dim, self.latent_dim, Cross=True)
        self.Cross_att2 = AttentionLayer(self.latent_dim, self.latent_dim, Cross=True)

    def forward(self, feats, modalitys, k1, k2, k3):
        img_idx = modalitys.view(-1) == 0
        txt_idx = modalitys.view(-1) == 1
        img_feats = feats[img_idx]
        txt_feats = feats[txt_idx]
        c_fea_img = torch.zeros(feats.shape[0], self.latent_dim).cuda()
        c_fea_txt = torch.zeros(feats.shape[0], self.latent_dim).cuda()

        imgs_recon, imgs_latent = self.imgAE(img_feats)
        txts_recon, txts_latent = self.txtAE(txt_feats)

        img2txt_recon, _ = self.img2txt(imgs_latent)
        img_latent_recon, _ = self.txt2img(img2txt_recon)
        img_feats_recon = self.imgAE.decoder(img_latent_recon)

        txt2img_recon, _ = self.txt2img(txts_latent)
        txt_latent_recon, _ = self.img2txt(txt2img_recon)
        txt_feats_recon = self.txtAE.decoder(txt_latent_recon)

        c_fea_img[img_idx] = imgs_latent

        c_fea_txt[txt_idx] = txts_latent

        # intra_modal complete
        txt_img_adj = self.img_att(txt2img_recon, imgs_latent, k1)
        g_fea_img = torch.mm(txt_img_adj, imgs_latent)
        c_fea_img[txt_idx] = g_fea_img

        img_txt_adj = self.txt_att(img2txt_recon, txts_latent, k1)
        g_fea_txt = torch.mm(img_txt_adj, txts_latent)
        c_fea_txt[img_idx] = g_fea_txt

        # intra_modal
        img_adj = built_cos_knn(c_fea_img, k2)
        c_fea_img = torch.mm(img_adj, c_fea_img)

        txt_adj = built_cos_knn(c_fea_txt, k2)
        c_fea_txt = torch.mm(txt_adj, c_fea_txt)

        # inter_modal
        S1, S2 = self.img_att1(c_fea_img, c_fea_txt, k3)
        c_txt2img_fea = torch.mm(S1, c_fea_txt)
        h_img_feat = c_fea_img + c_txt2img_fea

        c_img2txt_fea = torch.mm(S2, c_fea_img)
        h_txt_feat = c_fea_txt + c_img2txt_fea

        h_fuse = torch.cat((h_img_feat, h_txt_feat), dim=1)

        h_img_all = [
            img_feats,
            imgs_recon,
            imgs_latent,
            img_latent_recon,
            img_feats_recon,
        ]
        h_txt_all = [
            txt_feats,
            txts_recon,
            txts_latent,
            txt_latent_recon,
            txt_feats_recon,
        ]

        return h_img_all, h_txt_all, h_fuse, c_fea_img, c_fea_txt
