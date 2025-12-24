import torch

def save_ckpt(path, gen, mpd, msd, opt_g, opt_d, epoch, cfg):
    torch.save({
        "gen": gen.state_dict(),
        "mpd": mpd.state_dict(),
        "msd": msd.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "cfg": cfg,
    }, path)
