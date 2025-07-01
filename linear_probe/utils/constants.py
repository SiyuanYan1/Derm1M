IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]

MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "mae_base": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "dinov2": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "eva02_m38m": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "eva02_in22k": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "SwAVDerm": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "imgnet_base": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "imgnet_large": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "De_FM_base": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "De_FM_MILAN": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "MILAN": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "CAE_large": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "CAE_base": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
	"clip_base":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
	"clip_large":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	}
}