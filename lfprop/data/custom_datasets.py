import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from PIL import Image
from sklearn import datasets as skdata

LABEL_MAP_IMAGENET = {
    "n01440764": {"label": 0, "name": "tench"},
    "n01443537": {"label": 1, "name": "goldfish"},
    "n01484850": {"label": 2, "name": "great_white_shark"},
    "n01491361": {"label": 3, "name": "tiger_shark"},
    "n01494475": {"label": 4, "name": "hammerhead"},
    "n01496331": {"label": 5, "name": "electric_ray"},
    "n01498041": {"label": 6, "name": "stingray"},
    "n01514668": {"label": 7, "name": "cock"},
    "n01514859": {"label": 8, "name": "hen"},
    "n01518878": {"label": 9, "name": "ostrich"},
    "n01530575": {"label": 10, "name": "brambling"},
    "n01531178": {"label": 11, "name": "goldfinch"},
    "n01532829": {"label": 12, "name": "house_finch"},
    "n01534433": {"label": 13, "name": "junco"},
    "n01537544": {"label": 14, "name": "indigo_bunting"},
    "n01558993": {"label": 15, "name": "robin"},
    "n01560419": {"label": 16, "name": "bulbul"},
    "n01580077": {"label": 17, "name": "jay"},
    "n01582220": {"label": 18, "name": "magpie"},
    "n01592084": {"label": 19, "name": "chickadee"},
    "n01601694": {"label": 20, "name": "water_ouzel"},
    "n01608432": {"label": 21, "name": "kite"},
    "n01614925": {"label": 22, "name": "bald_eagle"},
    "n01616318": {"label": 23, "name": "vulture"},
    "n01622779": {"label": 24, "name": "great_grey_owl"},
    "n01629819": {"label": 25, "name": "European_fire_salamander"},
    "n01630670": {"label": 26, "name": "common_newt"},
    "n01631663": {"label": 27, "name": "eft"},
    "n01632458": {"label": 28, "name": "spotted_salamander"},
    "n01632777": {"label": 29, "name": "axolotl"},
    "n01641577": {"label": 30, "name": "bullfrog"},
    "n01644373": {"label": 31, "name": "tree_frog"},
    "n01644900": {"label": 32, "name": "tailed_frog"},
    "n01664065": {"label": 33, "name": "loggerhead"},
    "n01665541": {"label": 34, "name": "leatherback_turtle"},
    "n01667114": {"label": 35, "name": "mud_turtle"},
    "n01667778": {"label": 36, "name": "terrapin"},
    "n01669191": {"label": 37, "name": "box_turtle"},
    "n01675722": {"label": 38, "name": "banded_gecko"},
    "n01677366": {"label": 39, "name": "common_iguana"},
    "n01682714": {"label": 40, "name": "American_chameleon"},
    "n01685808": {"label": 41, "name": "whiptail"},
    "n01687978": {"label": 42, "name": "agama"},
    "n01688243": {"label": 43, "name": "frilled_lizard"},
    "n01689811": {"label": 44, "name": "alligator_lizard"},
    "n01692333": {"label": 45, "name": "Gila_monster"},
    "n01693334": {"label": 46, "name": "green_lizard"},
    "n01694178": {"label": 47, "name": "African_chameleon"},
    "n01695060": {"label": 48, "name": "Komodo_dragon"},
    "n01697457": {"label": 49, "name": "African_crocodile"},
    "n01698640": {"label": 50, "name": "American_alligator"},
    "n01704323": {"label": 51, "name": "triceratops"},
    "n01728572": {"label": 52, "name": "thunder_snake"},
    "n01728920": {"label": 53, "name": "ringneck_snake"},
    "n01729322": {"label": 54, "name": "hognose_snake"},
    "n01729977": {"label": 55, "name": "green_snake"},
    "n01734418": {"label": 56, "name": "king_snake"},
    "n01735189": {"label": 57, "name": "garter_snake"},
    "n01737021": {"label": 58, "name": "water_snake"},
    "n01739381": {"label": 59, "name": "vine_snake"},
    "n01740131": {"label": 60, "name": "night_snake"},
    "n01742172": {"label": 61, "name": "boa_constrictor"},
    "n01744401": {"label": 62, "name": "rock_python"},
    "n01748264": {"label": 63, "name": "Indian_cobra"},
    "n01749939": {"label": 64, "name": "green_mamba"},
    "n01751748": {"label": 65, "name": "sea_snake"},
    "n01753488": {"label": 66, "name": "horned_viper"},
    "n01755581": {"label": 67, "name": "diamondback"},
    "n01756291": {"label": 68, "name": "sidewinder"},
    "n01768244": {"label": 69, "name": "trilobite"},
    "n01770081": {"label": 70, "name": "harvestman"},
    "n01770393": {"label": 71, "name": "scorpion"},
    "n01773157": {"label": 72, "name": "black_and_gold_garden_spider"},
    "n01773549": {"label": 73, "name": "barn_spider"},
    "n01773797": {"label": 74, "name": "garden_spider"},
    "n01774384": {"label": 75, "name": "black_widow"},
    "n01774750": {"label": 76, "name": "tarantula"},
    "n01775062": {"label": 77, "name": "wolf_spider"},
    "n01776313": {"label": 78, "name": "tick"},
    "n01784675": {"label": 79, "name": "centipede"},
    "n01795545": {"label": 80, "name": "black_grouse"},
    "n01796340": {"label": 81, "name": "ptarmigan"},
    "n01797886": {"label": 82, "name": "ruffed_grouse"},
    "n01798484": {"label": 83, "name": "prairie_chicken"},
    "n01806143": {"label": 84, "name": "peacock"},
    "n01806567": {"label": 85, "name": "quail"},
    "n01807496": {"label": 86, "name": "partridge"},
    "n01817953": {"label": 87, "name": "African_grey"},
    "n01818515": {"label": 88, "name": "macaw"},
    "n01819313": {"label": 89, "name": "sulphur-crested_cockatoo"},
    "n01820546": {"label": 90, "name": "lorikeet"},
    "n01824575": {"label": 91, "name": "coucal"},
    "n01828970": {"label": 92, "name": "bee_eater"},
    "n01829413": {"label": 93, "name": "hornbill"},
    "n01833805": {"label": 94, "name": "hummingbird"},
    "n01843065": {"label": 95, "name": "jacamar"},
    "n01843383": {"label": 96, "name": "toucan"},
    "n01847000": {"label": 97, "name": "drake"},
    "n01855032": {"label": 98, "name": "red-breasted_merganser"},
    "n01855672": {"label": 99, "name": "goose"},
    "n01860187": {"label": 100, "name": "black_swan"},
    "n01871265": {"label": 101, "name": "tusker"},
    "n01872401": {"label": 102, "name": "echidna"},
    "n01873310": {"label": 103, "name": "platypus"},
    "n01877812": {"label": 104, "name": "wallaby"},
    "n01882714": {"label": 105, "name": "koala"},
    "n01883070": {"label": 106, "name": "wombat"},
    "n01910747": {"label": 107, "name": "jellyfish"},
    "n01914609": {"label": 108, "name": "sea_anemone"},
    "n01917289": {"label": 109, "name": "brain_coral"},
    "n01924916": {"label": 110, "name": "flatworm"},
    "n01930112": {"label": 111, "name": "nematode"},
    "n01943899": {"label": 112, "name": "conch"},
    "n01944390": {"label": 113, "name": "snail"},
    "n01945685": {"label": 114, "name": "slug"},
    "n01950731": {"label": 115, "name": "sea_slug"},
    "n01955084": {"label": 116, "name": "chiton"},
    "n01968897": {"label": 117, "name": "chambered_nautilus"},
    "n01978287": {"label": 118, "name": "Dungeness_crab"},
    "n01978455": {"label": 119, "name": "rock_crab"},
    "n01980166": {"label": 120, "name": "fiddler_crab"},
    "n01981276": {"label": 121, "name": "king_crab"},
    "n01983481": {"label": 122, "name": "American_lobster"},
    "n01984695": {"label": 123, "name": "spiny_lobster"},
    "n01985128": {"label": 124, "name": "crayfish"},
    "n01986214": {"label": 125, "name": "hermit_crab"},
    "n01990800": {"label": 126, "name": "isopod"},
    "n02002556": {"label": 127, "name": "white_stork"},
    "n02002724": {"label": 128, "name": "black_stork"},
    "n02006656": {"label": 129, "name": "spoonbill"},
    "n02007558": {"label": 130, "name": "flamingo"},
    "n02009229": {"label": 131, "name": "little_blue_heron"},
    "n02009912": {"label": 132, "name": "American_egret"},
    "n02011460": {"label": 133, "name": "bittern"},
    "n02012849": {"label": 134, "name": "crane"},
    "n02013706": {"label": 135, "name": "limpkin"},
    "n02017213": {"label": 136, "name": "European_gallinule"},
    "n02018207": {"label": 137, "name": "American_coot"},
    "n02018795": {"label": 138, "name": "bustard"},
    "n02025239": {"label": 139, "name": "ruddy_turnstone"},
    "n02027492": {"label": 140, "name": "red-backed_sandpiper"},
    "n02028035": {"label": 141, "name": "redshank"},
    "n02033041": {"label": 142, "name": "dowitcher"},
    "n02037110": {"label": 143, "name": "oystercatcher"},
    "n02051845": {"label": 144, "name": "pelican"},
    "n02056570": {"label": 145, "name": "king_penguin"},
    "n02058221": {"label": 146, "name": "albatross"},
    "n02066245": {"label": 147, "name": "grey_whale"},
    "n02071294": {"label": 148, "name": "killer_whale"},
    "n02074367": {"label": 149, "name": "dugong"},
    "n02077923": {"label": 150, "name": "sea_lion"},
    "n02085620": {"label": 151, "name": "Chihuahua"},
    "n02085782": {"label": 152, "name": "Japanese_spaniel"},
    "n02085936": {"label": 153, "name": "Maltese_dog"},
    "n02086079": {"label": 154, "name": "Pekinese"},
    "n02086240": {"label": 155, "name": "Shih-Tzu"},
    "n02086646": {"label": 156, "name": "Blenheim_spaniel"},
    "n02086910": {"label": 157, "name": "papillon"},
    "n02087046": {"label": 158, "name": "toy_terrier"},
    "n02087394": {"label": 159, "name": "Rhodesian_ridgeback"},
    "n02088094": {"label": 160, "name": "Afghan_hound"},
    "n02088238": {"label": 161, "name": "basset"},
    "n02088364": {"label": 162, "name": "beagle"},
    "n02088466": {"label": 163, "name": "bloodhound"},
    "n02088632": {"label": 164, "name": "bluetick"},
    "n02089078": {"label": 165, "name": "black-and-tan_coonhound"},
    "n02089867": {"label": 166, "name": "Walker_hound"},
    "n02089973": {"label": 167, "name": "English_foxhound"},
    "n02090379": {"label": 168, "name": "redbone"},
    "n02090622": {"label": 169, "name": "borzoi"},
    "n02090721": {"label": 170, "name": "Irish_wolfhound"},
    "n02091032": {"label": 171, "name": "Italian_greyhound"},
    "n02091134": {"label": 172, "name": "whippet"},
    "n02091244": {"label": 173, "name": "Ibizan_hound"},
    "n02091467": {"label": 174, "name": "Norwegian_elkhound"},
    "n02091635": {"label": 175, "name": "otterhound"},
    "n02091831": {"label": 176, "name": "Saluki"},
    "n02092002": {"label": 177, "name": "Scottish_deerhound"},
    "n02092339": {"label": 178, "name": "Weimaraner"},
    "n02093256": {"label": 179, "name": "Staffordshire_bullterrier"},
    "n02093428": {"label": 180, "name": "American_Staffordshire_terrier"},
    "n02093647": {"label": 181, "name": "Bedlington_terrier"},
    "n02093754": {"label": 182, "name": "Border_terrier"},
    "n02093859": {"label": 183, "name": "Kerry_blue_terrier"},
    "n02093991": {"label": 184, "name": "Irish_terrier"},
    "n02094114": {"label": 185, "name": "Norfolk_terrier"},
    "n02094258": {"label": 186, "name": "Norwich_terrier"},
    "n02094433": {"label": 187, "name": "Yorkshire_terrier"},
    "n02095314": {"label": 188, "name": "wire-haired_fox_terrier"},
    "n02095570": {"label": 189, "name": "Lakeland_terrier"},
    "n02095889": {"label": 190, "name": "Sealyham_terrier"},
    "n02096051": {"label": 191, "name": "Airedale"},
    "n02096177": {"label": 192, "name": "cairn"},
    "n02096294": {"label": 193, "name": "Australian_terrier"},
    "n02096437": {"label": 194, "name": "Dandie_Dinmont"},
    "n02096585": {"label": 195, "name": "Boston_bull"},
    "n02097047": {"label": 196, "name": "miniature_schnauzer"},
    "n02097130": {"label": 197, "name": "giant_schnauzer"},
    "n02097209": {"label": 198, "name": "standard_schnauzer"},
    "n02097298": {"label": 199, "name": "Scotch_terrier"},
    "n02097474": {"label": 200, "name": "Tibetan_terrier"},
    "n02097658": {"label": 201, "name": "silky_terrier"},
    "n02098105": {"label": 202, "name": "soft-coated_wheaten_terrier"},
    "n02098286": {"label": 203, "name": "West_Highland_white_terrier"},
    "n02098413": {"label": 204, "name": "Lhasa"},
    "n02099267": {"label": 205, "name": "flat-coated_retriever"},
    "n02099429": {"label": 206, "name": "curly-coated_retriever"},
    "n02099601": {"label": 207, "name": "golden_retriever"},
    "n02099712": {"label": 208, "name": "Labrador_retriever"},
    "n02099849": {"label": 209, "name": "Chesapeake_Bay_retriever"},
    "n02100236": {"label": 210, "name": "German_short-haired_pointer"},
    "n02100583": {"label": 211, "name": "vizsla"},
    "n02100735": {"label": 212, "name": "English_setter"},
    "n02100877": {"label": 213, "name": "Irish_setter"},
    "n02101006": {"label": 214, "name": "Gordon_setter"},
    "n02101388": {"label": 215, "name": "Brittany_spaniel"},
    "n02101556": {"label": 216, "name": "clumber"},
    "n02102040": {"label": 217, "name": "English_springer"},
    "n02102177": {"label": 218, "name": "Welsh_springer_spaniel"},
    "n02102318": {"label": 219, "name": "cocker_spaniel"},
    "n02102480": {"label": 220, "name": "Sussex_spaniel"},
    "n02102973": {"label": 221, "name": "Irish_water_spaniel"},
    "n02104029": {"label": 222, "name": "kuvasz"},
    "n02104365": {"label": 223, "name": "schipperke"},
    "n02105056": {"label": 224, "name": "groenendael"},
    "n02105162": {"label": 225, "name": "malinois"},
    "n02105251": {"label": 226, "name": "briard"},
    "n02105412": {"label": 227, "name": "kelpie"},
    "n02105505": {"label": 228, "name": "komondor"},
    "n02105641": {"label": 229, "name": "Old_English_sheepdog"},
    "n02105855": {"label": 230, "name": "Shetland_sheepdog"},
    "n02106030": {"label": 231, "name": "collie"},
    "n02106166": {"label": 232, "name": "Border_collie"},
    "n02106382": {"label": 233, "name": "Bouvier_des_Flandres"},
    "n02106550": {"label": 234, "name": "Rottweiler"},
    "n02106662": {"label": 235, "name": "German_shepherd"},
    "n02107142": {"label": 236, "name": "Doberman"},
    "n02107312": {"label": 237, "name": "miniature_pinscher"},
    "n02107574": {"label": 238, "name": "Greater_Swiss_Mountain_dog"},
    "n02107683": {"label": 239, "name": "Bernese_mountain_dog"},
    "n02107908": {"label": 240, "name": "Appenzeller"},
    "n02108000": {"label": 241, "name": "EntleBucher"},
    "n02108089": {"label": 242, "name": "boxer"},
    "n02108422": {"label": 243, "name": "bull_mastiff"},
    "n02108551": {"label": 244, "name": "Tibetan_mastiff"},
    "n02108915": {"label": 245, "name": "French_bulldog"},
    "n02109047": {"label": 246, "name": "Great_Dane"},
    "n02109525": {"label": 247, "name": "Saint_Bernard"},
    "n02109961": {"label": 248, "name": "Eskimo_dog"},
    "n02110063": {"label": 249, "name": "malamute"},
    "n02110185": {"label": 250, "name": "Siberian_husky"},
    "n02110341": {"label": 251, "name": "dalmatian"},
    "n02110627": {"label": 252, "name": "affenpinscher"},
    "n02110806": {"label": 253, "name": "basenji"},
    "n02110958": {"label": 254, "name": "pug"},
    "n02111129": {"label": 255, "name": "Leonberg"},
    "n02111277": {"label": 256, "name": "Newfoundland"},
    "n02111500": {"label": 257, "name": "Great_Pyrenees"},
    "n02111889": {"label": 258, "name": "Samoyed"},
    "n02112018": {"label": 259, "name": "Pomeranian"},
    "n02112137": {"label": 260, "name": "chow"},
    "n02112350": {"label": 261, "name": "keeshond"},
    "n02112706": {"label": 262, "name": "Brabancon_griffon"},
    "n02113023": {"label": 263, "name": "Pembroke"},
    "n02113186": {"label": 264, "name": "Cardigan"},
    "n02113624": {"label": 265, "name": "toy_poodle"},
    "n02113712": {"label": 266, "name": "miniature_poodle"},
    "n02113799": {"label": 267, "name": "standard_poodle"},
    "n02113978": {"label": 268, "name": "Mexican_hairless"},
    "n02114367": {"label": 269, "name": "timber_wolf"},
    "n02114548": {"label": 270, "name": "white_wolf"},
    "n02114712": {"label": 271, "name": "red_wolf"},
    "n02114855": {"label": 272, "name": "coyote"},
    "n02115641": {"label": 273, "name": "dingo"},
    "n02115913": {"label": 274, "name": "dhole"},
    "n02116738": {"label": 275, "name": "African_hunting_dog"},
    "n02117135": {"label": 276, "name": "hyena"},
    "n02119022": {"label": 277, "name": "red_fox"},
    "n02119789": {"label": 278, "name": "kit_fox"},
    "n02120079": {"label": 279, "name": "Arctic_fox"},
    "n02120505": {"label": 280, "name": "grey_fox"},
    "n02123045": {"label": 281, "name": "tabby"},
    "n02123159": {"label": 282, "name": "tiger_cat"},
    "n02123394": {"label": 283, "name": "Persian_cat"},
    "n02123597": {"label": 284, "name": "Siamese_cat"},
    "n02124075": {"label": 285, "name": "Egyptian_cat"},
    "n02125311": {"label": 286, "name": "cougar"},
    "n02127052": {"label": 287, "name": "lynx"},
    "n02128385": {"label": 288, "name": "leopard"},
    "n02128757": {"label": 289, "name": "snow_leopard"},
    "n02128925": {"label": 290, "name": "jaguar"},
    "n02129165": {"label": 291, "name": "lion"},
    "n02129604": {"label": 292, "name": "tiger"},
    "n02130308": {"label": 293, "name": "cheetah"},
    "n02132136": {"label": 294, "name": "brown_bear"},
    "n02133161": {"label": 295, "name": "American_black_bear"},
    "n02134084": {"label": 296, "name": "ice_bear"},
    "n02134418": {"label": 297, "name": "sloth_bear"},
    "n02137549": {"label": 298, "name": "mongoose"},
    "n02138441": {"label": 299, "name": "meerkat"},
    "n02165105": {"label": 300, "name": "tiger_beetle"},
    "n02165456": {"label": 301, "name": "ladybug"},
    "n02167151": {"label": 302, "name": "ground_beetle"},
    "n02168699": {"label": 303, "name": "long-horned_beetle"},
    "n02169497": {"label": 304, "name": "leaf_beetle"},
    "n02172182": {"label": 305, "name": "dung_beetle"},
    "n02174001": {"label": 306, "name": "rhinoceros_beetle"},
    "n02177972": {"label": 307, "name": "weevil"},
    "n02190166": {"label": 308, "name": "fly"},
    "n02206856": {"label": 309, "name": "bee"},
    "n02219486": {"label": 310, "name": "ant"},
    "n02226429": {"label": 311, "name": "grasshopper"},
    "n02229544": {"label": 312, "name": "cricket"},
    "n02231487": {"label": 313, "name": "walking_stick"},
    "n02233338": {"label": 314, "name": "cockroach"},
    "n02236044": {"label": 315, "name": "mantis"},
    "n02256656": {"label": 316, "name": "cicada"},
    "n02259212": {"label": 317, "name": "leafhopper"},
    "n02264363": {"label": 318, "name": "lacewing"},
    "n02268443": {"label": 319, "name": "dragonfly"},
    "n02268853": {"label": 320, "name": "damselfly"},
    "n02276258": {"label": 321, "name": "admiral"},
    "n02277742": {"label": 322, "name": "ringlet"},
    "n02279972": {"label": 323, "name": "monarch"},
    "n02280649": {"label": 324, "name": "cabbage_butterfly"},
    "n02281406": {"label": 325, "name": "sulphur_butterfly"},
    "n02281787": {"label": 326, "name": "lycaenid"},
    "n02317335": {"label": 327, "name": "starfish"},
    "n02319095": {"label": 328, "name": "sea_urchin"},
    "n02321529": {"label": 329, "name": "sea_cucumber"},
    "n02325366": {"label": 330, "name": "wood_rabbit"},
    "n02326432": {"label": 331, "name": "hare"},
    "n02328150": {"label": 332, "name": "Angora"},
    "n02342885": {"label": 333, "name": "hamster"},
    "n02346627": {"label": 334, "name": "porcupine"},
    "n02356798": {"label": 335, "name": "fox_squirrel"},
    "n02361337": {"label": 336, "name": "marmot"},
    "n02363005": {"label": 337, "name": "beaver"},
    "n02364673": {"label": 338, "name": "guinea_pig"},
    "n02389026": {"label": 339, "name": "sorrel"},
    "n02391049": {"label": 340, "name": "zebra"},
    "n02395406": {"label": 341, "name": "hog"},
    "n02396427": {"label": 342, "name": "wild_boar"},
    "n02397096": {"label": 343, "name": "warthog"},
    "n02398521": {"label": 344, "name": "hippopotamus"},
    "n02403003": {"label": 345, "name": "ox"},
    "n02408429": {"label": 346, "name": "water_buffalo"},
    "n02410509": {"label": 347, "name": "bison"},
    "n02412080": {"label": 348, "name": "ram"},
    "n02415577": {"label": 349, "name": "bighorn"},
    "n02417914": {"label": 350, "name": "ibex"},
    "n02422106": {"label": 351, "name": "hartebeest"},
    "n02422699": {"label": 352, "name": "impala"},
    "n02423022": {"label": 353, "name": "gazelle"},
    "n02437312": {"label": 354, "name": "Arabian_camel"},
    "n02437616": {"label": 355, "name": "llama"},
    "n02441942": {"label": 356, "name": "weasel"},
    "n02442845": {"label": 357, "name": "mink"},
    "n02443114": {"label": 358, "name": "polecat"},
    "n02443484": {"label": 359, "name": "black-footed_ferret"},
    "n02444819": {"label": 360, "name": "otter"},
    "n02445715": {"label": 361, "name": "skunk"},
    "n02447366": {"label": 362, "name": "badger"},
    "n02454379": {"label": 363, "name": "armadillo"},
    "n02457408": {"label": 364, "name": "three-toed_sloth"},
    "n02480495": {"label": 365, "name": "orangutan"},
    "n02480855": {"label": 366, "name": "gorilla"},
    "n02481823": {"label": 367, "name": "chimpanzee"},
    "n02483362": {"label": 368, "name": "gibbon"},
    "n02483708": {"label": 369, "name": "siamang"},
    "n02484975": {"label": 370, "name": "guenon"},
    "n02486261": {"label": 371, "name": "patas"},
    "n02486410": {"label": 372, "name": "baboon"},
    "n02487347": {"label": 373, "name": "macaque"},
    "n02488291": {"label": 374, "name": "langur"},
    "n02488702": {"label": 375, "name": "colobus"},
    "n02489166": {"label": 376, "name": "proboscis_monkey"},
    "n02490219": {"label": 377, "name": "marmoset"},
    "n02492035": {"label": 378, "name": "capuchin"},
    "n02492660": {"label": 379, "name": "howler_monkey"},
    "n02493509": {"label": 380, "name": "titi"},
    "n02493793": {"label": 381, "name": "spider_monkey"},
    "n02494079": {"label": 382, "name": "squirrel_monkey"},
    "n02497673": {"label": 383, "name": "Madagascar_cat"},
    "n02500267": {"label": 384, "name": "indri"},
    "n02504013": {"label": 385, "name": "Indian_elephant"},
    "n02504458": {"label": 386, "name": "African_elephant"},
    "n02509815": {"label": 387, "name": "lesser_panda"},
    "n02510455": {"label": 388, "name": "giant_panda"},
    "n02514041": {"label": 389, "name": "barracouta"},
    "n02526121": {"label": 390, "name": "eel"},
    "n02536864": {"label": 391, "name": "coho"},
    "n02606052": {"label": 392, "name": "rock_beauty"},
    "n02607072": {"label": 393, "name": "anemone_fish"},
    "n02640242": {"label": 394, "name": "sturgeon"},
    "n02641379": {"label": 395, "name": "gar"},
    "n02643566": {"label": 396, "name": "lionfish"},
    "n02655020": {"label": 397, "name": "puffer"},
    "n02666196": {"label": 398, "name": "abacus"},
    "n02667093": {"label": 399, "name": "abaya"},
    "n02669723": {"label": 400, "name": "academic_gown"},
    "n02672831": {"label": 401, "name": "accordion"},
    "n02676566": {"label": 402, "name": "acoustic_guitar"},
    "n02687172": {"label": 403, "name": "aircraft_carrier"},
    "n02690373": {"label": 404, "name": "airliner"},
    "n02692877": {"label": 405, "name": "airship"},
    "n02699494": {"label": 406, "name": "altar"},
    "n02701002": {"label": 407, "name": "ambulance"},
    "n02704792": {"label": 408, "name": "amphibian"},
    "n02708093": {"label": 409, "name": "analog_clock"},
    "n02727426": {"label": 410, "name": "apiary"},
    "n02730930": {"label": 411, "name": "apron"},
    "n02747177": {"label": 412, "name": "ashcan"},
    "n02749479": {"label": 413, "name": "assault_rifle"},
    "n02769748": {"label": 414, "name": "backpack"},
    "n02776631": {"label": 415, "name": "bakery"},
    "n02777292": {"label": 416, "name": "balance_beam"},
    "n02782093": {"label": 417, "name": "balloon"},
    "n02783161": {"label": 418, "name": "ballpoint"},
    "n02786058": {"label": 419, "name": "Band_Aid"},
    "n02787622": {"label": 420, "name": "banjo"},
    "n02788148": {"label": 421, "name": "bannister"},
    "n02790996": {"label": 422, "name": "barbell"},
    "n02791124": {"label": 423, "name": "barber_chair"},
    "n02791270": {"label": 424, "name": "barbershop"},
    "n02793495": {"label": 425, "name": "barn"},
    "n02794156": {"label": 426, "name": "barometer"},
    "n02795169": {"label": 427, "name": "barrel"},
    "n02797295": {"label": 428, "name": "barrow"},
    "n02799071": {"label": 429, "name": "baseball"},
    "n02802426": {"label": 430, "name": "basketball"},
    "n02804414": {"label": 431, "name": "bassinet"},
    "n02804610": {"label": 432, "name": "bassoon"},
    "n02807133": {"label": 433, "name": "bathing_cap"},
    "n02808304": {"label": 434, "name": "bath_towel"},
    "n02808440": {"label": 435, "name": "bathtub"},
    "n02814533": {"label": 436, "name": "beach_wagon"},
    "n02814860": {"label": 437, "name": "beacon"},
    "n02815834": {"label": 438, "name": "beaker"},
    "n02817516": {"label": 439, "name": "bearskin"},
    "n02823428": {"label": 440, "name": "beer_bottle"},
    "n02823750": {"label": 441, "name": "beer_glass"},
    "n02825657": {"label": 442, "name": "bell_cote"},
    "n02834397": {"label": 443, "name": "bib"},
    "n02835271": {"label": 444, "name": "bicycle-built-for-two"},
    "n02837789": {"label": 445, "name": "bikini"},
    "n02840245": {"label": 446, "name": "binder"},
    "n02841315": {"label": 447, "name": "binoculars"},
    "n02843684": {"label": 448, "name": "birdhouse"},
    "n02859443": {"label": 449, "name": "boathouse"},
    "n02860847": {"label": 450, "name": "bobsled"},
    "n02865351": {"label": 451, "name": "bolo_tie"},
    "n02869837": {"label": 452, "name": "bonnet"},
    "n02870880": {"label": 453, "name": "bookcase"},
    "n02871525": {"label": 454, "name": "bookshop"},
    "n02877765": {"label": 455, "name": "bottlecap"},
    "n02879718": {"label": 456, "name": "bow"},
    "n02883205": {"label": 457, "name": "bow_tie"},
    "n02892201": {"label": 458, "name": "brass"},
    "n02892767": {"label": 459, "name": "brassiere"},
    "n02894605": {"label": 460, "name": "breakwater"},
    "n02895154": {"label": 461, "name": "breastplate"},
    "n02906734": {"label": 462, "name": "broom"},
    "n02909870": {"label": 463, "name": "bucket"},
    "n02910353": {"label": 464, "name": "buckle"},
    "n02916936": {"label": 465, "name": "bulletproof_vest"},
    "n02917067": {"label": 466, "name": "bullet_train"},
    "n02927161": {"label": 467, "name": "butcher_shop"},
    "n02930766": {"label": 468, "name": "cab"},
    "n02939185": {"label": 469, "name": "caldron"},
    "n02948072": {"label": 470, "name": "candle"},
    "n02950826": {"label": 471, "name": "cannon"},
    "n02951358": {"label": 472, "name": "canoe"},
    "n02951585": {"label": 473, "name": "can_opener"},
    "n02963159": {"label": 474, "name": "cardigan"},
    "n02965783": {"label": 475, "name": "car_mirror"},
    "n02966193": {"label": 476, "name": "carousel"},
    "n02966687": {"label": 477, "name": "carpenter's_kit"},
    "n02971356": {"label": 478, "name": "carton"},
    "n02974003": {"label": 479, "name": "car_wheel"},
    "n02977058": {"label": 480, "name": "cash_machine"},
    "n02978881": {"label": 481, "name": "cassette"},
    "n02979186": {"label": 482, "name": "cassette_player"},
    "n02980441": {"label": 483, "name": "castle"},
    "n02981792": {"label": 484, "name": "catamaran"},
    "n02988304": {"label": 485, "name": "CD_player"},
    "n02992211": {"label": 486, "name": "cello"},
    "n02992529": {"label": 487, "name": "cellular_telephone"},
    "n02999410": {"label": 488, "name": "chain"},
    "n03000134": {"label": 489, "name": "chainlink_fence"},
    "n03000247": {"label": 490, "name": "chain_mail"},
    "n03000684": {"label": 491, "name": "chain_saw"},
    "n03014705": {"label": 492, "name": "chest"},
    "n03016953": {"label": 493, "name": "chiffonier"},
    "n03017168": {"label": 494, "name": "chime"},
    "n03018349": {"label": 495, "name": "china_cabinet"},
    "n03026506": {"label": 496, "name": "Christmas_stocking"},
    "n03028079": {"label": 497, "name": "church"},
    "n03032252": {"label": 498, "name": "cinema"},
    "n03041632": {"label": 499, "name": "cleaver"},
    "n03042490": {"label": 500, "name": "cliff_dwelling"},
    "n03045698": {"label": 501, "name": "cloak"},
    "n03047690": {"label": 502, "name": "clog"},
    "n03062245": {"label": 503, "name": "cocktail_shaker"},
    "n03063599": {"label": 504, "name": "coffee_mug"},
    "n03063689": {"label": 505, "name": "coffeepot"},
    "n03065424": {"label": 506, "name": "coil"},
    "n03075370": {"label": 507, "name": "combination_lock"},
    "n03085013": {"label": 508, "name": "computer_keyboard"},
    "n03089624": {"label": 509, "name": "confectionery"},
    "n03095699": {"label": 510, "name": "container_ship"},
    "n03100240": {"label": 511, "name": "convertible"},
    "n03109150": {"label": 512, "name": "corkscrew"},
    "n03110669": {"label": 513, "name": "cornet"},
    "n03124043": {"label": 514, "name": "cowboy_boot"},
    "n03124170": {"label": 515, "name": "cowboy_hat"},
    "n03125729": {"label": 516, "name": "cradle"},
    "n03126707": {"label": 517, "name": "crane"},
    "n03127747": {"label": 518, "name": "crash_helmet"},
    "n03127925": {"label": 519, "name": "crate"},
    "n03131574": {"label": 520, "name": "crib"},
    "n03133878": {"label": 521, "name": "Crock_Pot"},
    "n03134739": {"label": 522, "name": "croquet_ball"},
    "n03141823": {"label": 523, "name": "crutch"},
    "n03146219": {"label": 524, "name": "cuirass"},
    "n03160309": {"label": 525, "name": "dam"},
    "n03179701": {"label": 526, "name": "desk"},
    "n03180011": {"label": 527, "name": "desktop_computer"},
    "n03187595": {"label": 528, "name": "dial_telephone"},
    "n03188531": {"label": 529, "name": "diaper"},
    "n03196217": {"label": 530, "name": "digital_clock"},
    "n03197337": {"label": 531, "name": "digital_watch"},
    "n03201208": {"label": 532, "name": "dining_table"},
    "n03207743": {"label": 533, "name": "dishrag"},
    "n03207941": {"label": 534, "name": "dishwasher"},
    "n03208938": {"label": 535, "name": "disk_brake"},
    "n03216828": {"label": 536, "name": "dock"},
    "n03218198": {"label": 537, "name": "dogsled"},
    "n03220513": {"label": 538, "name": "dome"},
    "n03223299": {"label": 539, "name": "doormat"},
    "n03240683": {"label": 540, "name": "drilling_platform"},
    "n03249569": {"label": 541, "name": "drum"},
    "n03250847": {"label": 542, "name": "drumstick"},
    "n03255030": {"label": 543, "name": "dumbbell"},
    "n03259280": {"label": 544, "name": "Dutch_oven"},
    "n03271574": {"label": 545, "name": "electric_fan"},
    "n03272010": {"label": 546, "name": "electric_guitar"},
    "n03272562": {"label": 547, "name": "electric_locomotive"},
    "n03290653": {"label": 548, "name": "entertainment_center"},
    "n03291819": {"label": 549, "name": "envelope"},
    "n03297495": {"label": 550, "name": "espresso_maker"},
    "n03314780": {"label": 551, "name": "face_powder"},
    "n03325584": {"label": 552, "name": "feather_boa"},
    "n03337140": {"label": 553, "name": "file"},
    "n03344393": {"label": 554, "name": "fireboat"},
    "n03345487": {"label": 555, "name": "fire_engine"},
    "n03347037": {"label": 556, "name": "fire_screen"},
    "n03355925": {"label": 557, "name": "flagpole"},
    "n03372029": {"label": 558, "name": "flute"},
    "n03376595": {"label": 559, "name": "folding_chair"},
    "n03379051": {"label": 560, "name": "football_helmet"},
    "n03384352": {"label": 561, "name": "forklift"},
    "n03388043": {"label": 562, "name": "fountain"},
    "n03388183": {"label": 563, "name": "fountain_pen"},
    "n03388549": {"label": 564, "name": "four-poster"},
    "n03393912": {"label": 565, "name": "freight_car"},
    "n03394916": {"label": 566, "name": "French_horn"},
    "n03400231": {"label": 567, "name": "frying_pan"},
    "n03404251": {"label": 568, "name": "fur_coat"},
    "n03417042": {"label": 569, "name": "garbage_truck"},
    "n03424325": {"label": 570, "name": "gasmask"},
    "n03425413": {"label": 571, "name": "gas_pump"},
    "n03443371": {"label": 572, "name": "goblet"},
    "n03444034": {"label": 573, "name": "go-kart"},
    "n03445777": {"label": 574, "name": "golf_ball"},
    "n03445924": {"label": 575, "name": "golfcart"},
    "n03447447": {"label": 576, "name": "gondola"},
    "n03447721": {"label": 577, "name": "gong"},
    "n03450230": {"label": 578, "name": "gown"},
    "n03452741": {"label": 579, "name": "grand_piano"},
    "n03457902": {"label": 580, "name": "greenhouse"},
    "n03459775": {"label": 581, "name": "grille"},
    "n03461385": {"label": 582, "name": "grocery_store"},
    "n03467068": {"label": 583, "name": "guillotine"},
    "n03476684": {"label": 584, "name": "hair_slide"},
    "n03476991": {"label": 585, "name": "hair_spray"},
    "n03478589": {"label": 586, "name": "half_track"},
    "n03481172": {"label": 587, "name": "hammer"},
    "n03482405": {"label": 588, "name": "hamper"},
    "n03483316": {"label": 589, "name": "hand_blower"},
    "n03485407": {"label": 590, "name": "hand-held_computer"},
    "n03485794": {"label": 591, "name": "handkerchief"},
    "n03492542": {"label": 592, "name": "hard_disc"},
    "n03494278": {"label": 593, "name": "harmonica"},
    "n03495258": {"label": 594, "name": "harp"},
    "n03496892": {"label": 595, "name": "harvester"},
    "n03498962": {"label": 596, "name": "hatchet"},
    "n03527444": {"label": 597, "name": "holster"},
    "n03529860": {"label": 598, "name": "home_theater"},
    "n03530642": {"label": 599, "name": "honeycomb"},
    "n03532672": {"label": 600, "name": "hook"},
    "n03534580": {"label": 601, "name": "hoopskirt"},
    "n03535780": {"label": 602, "name": "horizontal_bar"},
    "n03538406": {"label": 603, "name": "horse_cart"},
    "n03544143": {"label": 604, "name": "hourglass"},
    "n03584254": {"label": 605, "name": "iPod"},
    "n03584829": {"label": 606, "name": "iron"},
    "n03590841": {"label": 607, "name": "jack-o'-lantern"},
    "n03594734": {"label": 608, "name": "jean"},
    "n03594945": {"label": 609, "name": "jeep"},
    "n03595614": {"label": 610, "name": "jersey"},
    "n03598930": {"label": 611, "name": "jigsaw_puzzle"},
    "n03599486": {"label": 612, "name": "jinrikisha"},
    "n03602883": {"label": 613, "name": "joystick"},
    "n03617480": {"label": 614, "name": "kimono"},
    "n03623198": {"label": 615, "name": "knee_pad"},
    "n03627232": {"label": 616, "name": "knot"},
    "n03630383": {"label": 617, "name": "lab_coat"},
    "n03633091": {"label": 618, "name": "ladle"},
    "n03637318": {"label": 619, "name": "lampshade"},
    "n03642806": {"label": 620, "name": "laptop"},
    "n03649909": {"label": 621, "name": "lawn_mower"},
    "n03657121": {"label": 622, "name": "lens_cap"},
    "n03658185": {"label": 623, "name": "letter_opener"},
    "n03661043": {"label": 624, "name": "library"},
    "n03662601": {"label": 625, "name": "lifeboat"},
    "n03666591": {"label": 626, "name": "lighter"},
    "n03670208": {"label": 627, "name": "limousine"},
    "n03673027": {"label": 628, "name": "liner"},
    "n03676483": {"label": 629, "name": "lipstick"},
    "n03680355": {"label": 630, "name": "Loafer"},
    "n03690938": {"label": 631, "name": "lotion"},
    "n03691459": {"label": 632, "name": "loudspeaker"},
    "n03692522": {"label": 633, "name": "loupe"},
    "n03697007": {"label": 634, "name": "lumbermill"},
    "n03706229": {"label": 635, "name": "magnetic_compass"},
    "n03709823": {"label": 636, "name": "mailbag"},
    "n03710193": {"label": 637, "name": "mailbox"},
    "n03710637": {"label": 638, "name": "maillot"},
    "n03710721": {"label": 639, "name": "maillot"},
    "n03717622": {"label": 640, "name": "manhole_cover"},
    "n03720891": {"label": 641, "name": "maraca"},
    "n03721384": {"label": 642, "name": "marimba"},
    "n03724870": {"label": 643, "name": "mask"},
    "n03729826": {"label": 644, "name": "matchstick"},
    "n03733131": {"label": 645, "name": "maypole"},
    "n03733281": {"label": 646, "name": "maze"},
    "n03733805": {"label": 647, "name": "measuring_cup"},
    "n03742115": {"label": 648, "name": "medicine_chest"},
    "n03743016": {"label": 649, "name": "megalith"},
    "n03759954": {"label": 650, "name": "microphone"},
    "n03761084": {"label": 651, "name": "microwave"},
    "n03763968": {"label": 652, "name": "military_uniform"},
    "n03764736": {"label": 653, "name": "milk_can"},
    "n03769881": {"label": 654, "name": "minibus"},
    "n03770439": {"label": 655, "name": "miniskirt"},
    "n03770679": {"label": 656, "name": "minivan"},
    "n03773504": {"label": 657, "name": "missile"},
    "n03775071": {"label": 658, "name": "mitten"},
    "n03775546": {"label": 659, "name": "mixing_bowl"},
    "n03776460": {"label": 660, "name": "mobile_home"},
    "n03777568": {"label": 661, "name": "Model_T"},
    "n03777754": {"label": 662, "name": "modem"},
    "n03781244": {"label": 663, "name": "monastery"},
    "n03782006": {"label": 664, "name": "monitor"},
    "n03785016": {"label": 665, "name": "moped"},
    "n03786901": {"label": 666, "name": "mortar"},
    "n03787032": {"label": 667, "name": "mortarboard"},
    "n03788195": {"label": 668, "name": "mosque"},
    "n03788365": {"label": 669, "name": "mosquito_net"},
    "n03791053": {"label": 670, "name": "motor_scooter"},
    "n03792782": {"label": 671, "name": "mountain_bike"},
    "n03792972": {"label": 672, "name": "mountain_tent"},
    "n03793489": {"label": 673, "name": "mouse"},
    "n03794056": {"label": 674, "name": "mousetrap"},
    "n03796401": {"label": 675, "name": "moving_van"},
    "n03803284": {"label": 676, "name": "muzzle"},
    "n03804744": {"label": 677, "name": "nail"},
    "n03814639": {"label": 678, "name": "neck_brace"},
    "n03814906": {"label": 679, "name": "necklace"},
    "n03825788": {"label": 680, "name": "nipple"},
    "n03832673": {"label": 681, "name": "notebook"},
    "n03837869": {"label": 682, "name": "obelisk"},
    "n03838899": {"label": 683, "name": "oboe"},
    "n03840681": {"label": 684, "name": "ocarina"},
    "n03841143": {"label": 685, "name": "odometer"},
    "n03843555": {"label": 686, "name": "oil_filter"},
    "n03854065": {"label": 687, "name": "organ"},
    "n03857828": {"label": 688, "name": "oscilloscope"},
    "n03866082": {"label": 689, "name": "overskirt"},
    "n03868242": {"label": 690, "name": "oxcart"},
    "n03868863": {"label": 691, "name": "oxygen_mask"},
    "n03871628": {"label": 692, "name": "packet"},
    "n03873416": {"label": 693, "name": "paddle"},
    "n03874293": {"label": 694, "name": "paddlewheel"},
    "n03874599": {"label": 695, "name": "padlock"},
    "n03876231": {"label": 696, "name": "paintbrush"},
    "n03877472": {"label": 697, "name": "pajama"},
    "n03877845": {"label": 698, "name": "palace"},
    "n03884397": {"label": 699, "name": "panpipe"},
    "n03887697": {"label": 700, "name": "paper_towel"},
    "n03888257": {"label": 701, "name": "parachute"},
    "n03888605": {"label": 702, "name": "parallel_bars"},
    "n03891251": {"label": 703, "name": "park_bench"},
    "n03891332": {"label": 704, "name": "parking_meter"},
    "n03895866": {"label": 705, "name": "passenger_car"},
    "n03899768": {"label": 706, "name": "patio"},
    "n03902125": {"label": 707, "name": "pay-phone"},
    "n03903868": {"label": 708, "name": "pedestal"},
    "n03908618": {"label": 709, "name": "pencil_box"},
    "n03908714": {"label": 710, "name": "pencil_sharpener"},
    "n03916031": {"label": 711, "name": "perfume"},
    "n03920288": {"label": 712, "name": "Petri_dish"},
    "n03924679": {"label": 713, "name": "photocopier"},
    "n03929660": {"label": 714, "name": "pick"},
    "n03929855": {"label": 715, "name": "pickelhaube"},
    "n03930313": {"label": 716, "name": "picket_fence"},
    "n03930630": {"label": 717, "name": "pickup"},
    "n03933933": {"label": 718, "name": "pier"},
    "n03935335": {"label": 719, "name": "piggy_bank"},
    "n03937543": {"label": 720, "name": "pill_bottle"},
    "n03938244": {"label": 721, "name": "pillow"},
    "n03942813": {"label": 722, "name": "ping-pong_ball"},
    "n03944341": {"label": 723, "name": "pinwheel"},
    "n03947888": {"label": 724, "name": "pirate"},
    "n03950228": {"label": 725, "name": "pitcher"},
    "n03954731": {"label": 726, "name": "plane"},
    "n03956157": {"label": 727, "name": "planetarium"},
    "n03958227": {"label": 728, "name": "plastic_bag"},
    "n03961711": {"label": 729, "name": "plate_rack"},
    "n03967562": {"label": 730, "name": "plow"},
    "n03970156": {"label": 731, "name": "plunger"},
    "n03976467": {"label": 732, "name": "Polaroid_camera"},
    "n03976657": {"label": 733, "name": "pole"},
    "n03977966": {"label": 734, "name": "police_van"},
    "n03980874": {"label": 735, "name": "poncho"},
    "n03982430": {"label": 736, "name": "pool_table"},
    "n03983396": {"label": 737, "name": "pop_bottle"},
    "n03991062": {"label": 738, "name": "pot"},
    "n03992509": {"label": 739, "name": "potter's_wheel"},
    "n03995372": {"label": 740, "name": "power_drill"},
    "n03998194": {"label": 741, "name": "prayer_rug"},
    "n04004767": {"label": 742, "name": "printer"},
    "n04005630": {"label": 743, "name": "prison"},
    "n04008634": {"label": 744, "name": "projectile"},
    "n04009552": {"label": 745, "name": "projector"},
    "n04019541": {"label": 746, "name": "puck"},
    "n04023962": {"label": 747, "name": "punching_bag"},
    "n04026417": {"label": 748, "name": "purse"},
    "n04033901": {"label": 749, "name": "quill"},
    "n04033995": {"label": 750, "name": "quilt"},
    "n04037443": {"label": 751, "name": "racer"},
    "n04039381": {"label": 752, "name": "racket"},
    "n04040759": {"label": 753, "name": "radiator"},
    "n04041544": {"label": 754, "name": "radio"},
    "n04044716": {"label": 755, "name": "radio_telescope"},
    "n04049303": {"label": 756, "name": "rain_barrel"},
    "n04065272": {"label": 757, "name": "recreational_vehicle"},
    "n04067472": {"label": 758, "name": "reel"},
    "n04069434": {"label": 759, "name": "reflex_camera"},
    "n04070727": {"label": 760, "name": "refrigerator"},
    "n04074963": {"label": 761, "name": "remote_control"},
    "n04081281": {"label": 762, "name": "restaurant"},
    "n04086273": {"label": 763, "name": "revolver"},
    "n04090263": {"label": 764, "name": "rifle"},
    "n04099969": {"label": 765, "name": "rocking_chair"},
    "n04111531": {"label": 766, "name": "rotisserie"},
    "n04116512": {"label": 767, "name": "rubber_eraser"},
    "n04118538": {"label": 768, "name": "rugby_ball"},
    "n04118776": {"label": 769, "name": "rule"},
    "n04120489": {"label": 770, "name": "running_shoe"},
    "n04125021": {"label": 771, "name": "safe"},
    "n04127249": {"label": 772, "name": "safety_pin"},
    "n04131690": {"label": 773, "name": "saltshaker"},
    "n04133789": {"label": 774, "name": "sandal"},
    "n04136333": {"label": 775, "name": "sarong"},
    "n04141076": {"label": 776, "name": "sax"},
    "n04141327": {"label": 777, "name": "scabbard"},
    "n04141975": {"label": 778, "name": "scale"},
    "n04146614": {"label": 779, "name": "school_bus"},
    "n04147183": {"label": 780, "name": "schooner"},
    "n04149813": {"label": 781, "name": "scoreboard"},
    "n04152593": {"label": 782, "name": "screen"},
    "n04153751": {"label": 783, "name": "screw"},
    "n04154565": {"label": 784, "name": "screwdriver"},
    "n04162706": {"label": 785, "name": "seat_belt"},
    "n04179913": {"label": 786, "name": "sewing_machine"},
    "n04192698": {"label": 787, "name": "shield"},
    "n04200800": {"label": 788, "name": "shoe_shop"},
    "n04201297": {"label": 789, "name": "shoji"},
    "n04204238": {"label": 790, "name": "shopping_basket"},
    "n04204347": {"label": 791, "name": "shopping_cart"},
    "n04208210": {"label": 792, "name": "shovel"},
    "n04209133": {"label": 793, "name": "shower_cap"},
    "n04209239": {"label": 794, "name": "shower_curtain"},
    "n04228054": {"label": 795, "name": "ski"},
    "n04229816": {"label": 796, "name": "ski_mask"},
    "n04235860": {"label": 797, "name": "sleeping_bag"},
    "n04238763": {"label": 798, "name": "slide_rule"},
    "n04239074": {"label": 799, "name": "sliding_door"},
    "n04243546": {"label": 800, "name": "slot"},
    "n04251144": {"label": 801, "name": "snorkel"},
    "n04252077": {"label": 802, "name": "snowmobile"},
    "n04252225": {"label": 803, "name": "snowplow"},
    "n04254120": {"label": 804, "name": "soap_dispenser"},
    "n04254680": {"label": 805, "name": "soccer_ball"},
    "n04254777": {"label": 806, "name": "sock"},
    "n04258138": {"label": 807, "name": "solar_dish"},
    "n04259630": {"label": 808, "name": "sombrero"},
    "n04263257": {"label": 809, "name": "soup_bowl"},
    "n04264628": {"label": 810, "name": "space_bar"},
    "n04265275": {"label": 811, "name": "space_heater"},
    "n04266014": {"label": 812, "name": "space_shuttle"},
    "n04270147": {"label": 813, "name": "spatula"},
    "n04273569": {"label": 814, "name": "speedboat"},
    "n04275548": {"label": 815, "name": "spider_web"},
    "n04277352": {"label": 816, "name": "spindle"},
    "n04285008": {"label": 817, "name": "sports_car"},
    "n04286575": {"label": 818, "name": "spotlight"},
    "n04296562": {"label": 819, "name": "stage"},
    "n04310018": {"label": 820, "name": "steam_locomotive"},
    "n04311004": {"label": 821, "name": "steel_arch_bridge"},
    "n04311174": {"label": 822, "name": "steel_drum"},
    "n04317175": {"label": 823, "name": "stethoscope"},
    "n04325704": {"label": 824, "name": "stole"},
    "n04326547": {"label": 825, "name": "stone_wall"},
    "n04328186": {"label": 826, "name": "stopwatch"},
    "n04330267": {"label": 827, "name": "stove"},
    "n04332243": {"label": 828, "name": "strainer"},
    "n04335435": {"label": 829, "name": "streetcar"},
    "n04336792": {"label": 830, "name": "stretcher"},
    "n04344873": {"label": 831, "name": "studio_couch"},
    "n04346328": {"label": 832, "name": "stupa"},
    "n04347754": {"label": 833, "name": "submarine"},
    "n04350905": {"label": 834, "name": "suit"},
    "n04355338": {"label": 835, "name": "sundial"},
    "n04355933": {"label": 836, "name": "sunglass"},
    "n04356056": {"label": 837, "name": "sunglasses"},
    "n04357314": {"label": 838, "name": "sunscreen"},
    "n04366367": {"label": 839, "name": "suspension_bridge"},
    "n04367480": {"label": 840, "name": "swab"},
    "n04370456": {"label": 841, "name": "sweatshirt"},
    "n04371430": {"label": 842, "name": "swimming_trunks"},
    "n04371774": {"label": 843, "name": "swing"},
    "n04372370": {"label": 844, "name": "switch"},
    "n04376876": {"label": 845, "name": "syringe"},
    "n04380533": {"label": 846, "name": "table_lamp"},
    "n04389033": {"label": 847, "name": "tank"},
    "n04392985": {"label": 848, "name": "tape_player"},
    "n04398044": {"label": 849, "name": "teapot"},
    "n04399382": {"label": 850, "name": "teddy"},
    "n04404412": {"label": 851, "name": "television"},
    "n04409515": {"label": 852, "name": "tennis_ball"},
    "n04417672": {"label": 853, "name": "thatch"},
    "n04418357": {"label": 854, "name": "theater_curtain"},
    "n04423845": {"label": 855, "name": "thimble"},
    "n04428191": {"label": 856, "name": "thresher"},
    "n04429376": {"label": 857, "name": "throne"},
    "n04435653": {"label": 858, "name": "tile_roof"},
    "n04442312": {"label": 859, "name": "toaster"},
    "n04443257": {"label": 860, "name": "tobacco_shop"},
    "n04447861": {"label": 861, "name": "toilet_seat"},
    "n04456115": {"label": 862, "name": "torch"},
    "n04458633": {"label": 863, "name": "totem_pole"},
    "n04461696": {"label": 864, "name": "tow_truck"},
    "n04462240": {"label": 865, "name": "toyshop"},
    "n04465501": {"label": 866, "name": "tractor"},
    "n04467665": {"label": 867, "name": "trailer_truck"},
    "n04476259": {"label": 868, "name": "tray"},
    "n04479046": {"label": 869, "name": "trench_coat"},
    "n04482393": {"label": 870, "name": "tricycle"},
    "n04483307": {"label": 871, "name": "trimaran"},
    "n04485082": {"label": 872, "name": "tripod"},
    "n04486054": {"label": 873, "name": "triumphal_arch"},
    "n04487081": {"label": 874, "name": "trolleybus"},
    "n04487394": {"label": 875, "name": "trombone"},
    "n04493381": {"label": 876, "name": "tub"},
    "n04501370": {"label": 877, "name": "turnstile"},
    "n04505470": {"label": 878, "name": "typewriter_keyboard"},
    "n04507155": {"label": 879, "name": "umbrella"},
    "n04509417": {"label": 880, "name": "unicycle"},
    "n04515003": {"label": 881, "name": "upright"},
    "n04517823": {"label": 882, "name": "vacuum"},
    "n04522168": {"label": 883, "name": "vase"},
    "n04523525": {"label": 884, "name": "vault"},
    "n04525038": {"label": 885, "name": "velvet"},
    "n04525305": {"label": 886, "name": "vending_machine"},
    "n04532106": {"label": 887, "name": "vestment"},
    "n04532670": {"label": 888, "name": "viaduct"},
    "n04536866": {"label": 889, "name": "violin"},
    "n04540053": {"label": 890, "name": "volleyball"},
    "n04542943": {"label": 891, "name": "waffle_iron"},
    "n04548280": {"label": 892, "name": "wall_clock"},
    "n04548362": {"label": 893, "name": "wallet"},
    "n04550184": {"label": 894, "name": "wardrobe"},
    "n04552348": {"label": 895, "name": "warplane"},
    "n04553703": {"label": 896, "name": "washbasin"},
    "n04554684": {"label": 897, "name": "washer"},
    "n04557648": {"label": 898, "name": "water_bottle"},
    "n04560804": {"label": 899, "name": "water_jug"},
    "n04562935": {"label": 900, "name": "water_tower"},
    "n04579145": {"label": 901, "name": "whiskey_jug"},
    "n04579432": {"label": 902, "name": "whistle"},
    "n04584207": {"label": 903, "name": "wig"},
    "n04589890": {"label": 904, "name": "window_screen"},
    "n04590129": {"label": 905, "name": "window_shade"},
    "n04591157": {"label": 906, "name": "Windsor_tie"},
    "n04591713": {"label": 907, "name": "wine_bottle"},
    "n04592741": {"label": 908, "name": "wing"},
    "n04596742": {"label": 909, "name": "wok"},
    "n04597913": {"label": 910, "name": "wooden_spoon"},
    "n04599235": {"label": 911, "name": "wool"},
    "n04604644": {"label": 912, "name": "worm_fence"},
    "n04606251": {"label": 913, "name": "wreck"},
    "n04612504": {"label": 914, "name": "yawl"},
    "n04613696": {"label": 915, "name": "yurt"},
    "n06359193": {"label": 916, "name": "web_site"},
    "n06596364": {"label": 917, "name": "comic_book"},
    "n06785654": {"label": 918, "name": "crossword_puzzle"},
    "n06794110": {"label": 919, "name": "street_sign"},
    "n06874185": {"label": 920, "name": "traffic_light"},
    "n07248320": {"label": 921, "name": "book_jacket"},
    "n07565083": {"label": 922, "name": "menu"},
    "n07579787": {"label": 923, "name": "plate"},
    "n07583066": {"label": 924, "name": "guacamole"},
    "n07584110": {"label": 925, "name": "consomme"},
    "n07590611": {"label": 926, "name": "hot_pot"},
    "n07613480": {"label": 927, "name": "trifle"},
    "n07614500": {"label": 928, "name": "ice_cream"},
    "n07615774": {"label": 929, "name": "ice_lolly"},
    "n07684084": {"label": 930, "name": "French_loaf"},
    "n07693725": {"label": 931, "name": "bagel"},
    "n07695742": {"label": 932, "name": "pretzel"},
    "n07697313": {"label": 933, "name": "cheeseburger"},
    "n07697537": {"label": 934, "name": "hotdog"},
    "n07711569": {"label": 935, "name": "mashed_potato"},
    "n07714571": {"label": 936, "name": "head_cabbage"},
    "n07714990": {"label": 937, "name": "broccoli"},
    "n07715103": {"label": 938, "name": "cauliflower"},
    "n07716358": {"label": 939, "name": "zucchini"},
    "n07716906": {"label": 940, "name": "spaghetti_squash"},
    "n07717410": {"label": 941, "name": "acorn_squash"},
    "n07717556": {"label": 942, "name": "butternut_squash"},
    "n07718472": {"label": 943, "name": "cucumber"},
    "n07718747": {"label": 944, "name": "artichoke"},
    "n07720875": {"label": 945, "name": "bell_pepper"},
    "n07730033": {"label": 946, "name": "cardoon"},
    "n07734744": {"label": 947, "name": "mushroom"},
    "n07742313": {"label": 948, "name": "Granny_Smith"},
    "n07745940": {"label": 949, "name": "strawberry"},
    "n07747607": {"label": 950, "name": "orange"},
    "n07749582": {"label": 951, "name": "lemon"},
    "n07753113": {"label": 952, "name": "fig"},
    "n07753275": {"label": 953, "name": "pineapple"},
    "n07753592": {"label": 954, "name": "banana"},
    "n07754684": {"label": 955, "name": "jackfruit"},
    "n07760859": {"label": 956, "name": "custard_apple"},
    "n07768694": {"label": 957, "name": "pomegranate"},
    "n07802026": {"label": 958, "name": "hay"},
    "n07831146": {"label": 959, "name": "carbonara"},
    "n07836838": {"label": 960, "name": "chocolate_sauce"},
    "n07860988": {"label": 961, "name": "dough"},
    "n07871810": {"label": 962, "name": "meat_loaf"},
    "n07873807": {"label": 963, "name": "pizza"},
    "n07875152": {"label": 964, "name": "potpie"},
    "n07880968": {"label": 965, "name": "burrito"},
    "n07892512": {"label": 966, "name": "red_wine"},
    "n07920052": {"label": 967, "name": "espresso"},
    "n07930864": {"label": 968, "name": "cup"},
    "n07932039": {"label": 969, "name": "eggnog"},
    "n09193705": {"label": 970, "name": "alp"},
    "n09229709": {"label": 971, "name": "bubble"},
    "n09246464": {"label": 972, "name": "cliff"},
    "n09256479": {"label": 973, "name": "coral_reef"},
    "n09288635": {"label": 974, "name": "geyser"},
    "n09332890": {"label": 975, "name": "lakeside"},
    "n09399592": {"label": 976, "name": "promontory"},
    "n09421951": {"label": 977, "name": "sandbar"},
    "n09428293": {"label": 978, "name": "seashore"},
    "n09468604": {"label": 979, "name": "valley"},
    "n09472597": {"label": 980, "name": "volcano"},
    "n09835506": {"label": 981, "name": "ballplayer"},
    "n10148035": {"label": 982, "name": "groom"},
    "n10565667": {"label": 983, "name": "scuba_diver"},
    "n11879895": {"label": 984, "name": "rapeseed"},
    "n11939491": {"label": 985, "name": "daisy"},
    "n12057211": {"label": 986, "name": "yellow_lady's_slipper"},
    "n12144580": {"label": 987, "name": "corn"},
    "n12267677": {"label": 988, "name": "acorn"},
    "n12620546": {"label": 989, "name": "hip"},
    "n12768682": {"label": 990, "name": "buckeye"},
    "n12985857": {"label": 991, "name": "coral_fungus"},
    "n12998815": {"label": 992, "name": "agaric"},
    "n13037406": {"label": 993, "name": "gyromitra"},
    "n13040303": {"label": 994, "name": "stinkhorn"},
    "n13044778": {"label": 995, "name": "earthstar"},
    "n13052670": {"label": 996, "name": "hen-of-the-woods"},
    "n13054560": {"label": 997, "name": "bolete"},
    "n13133613": {"label": 998, "name": "ear"},
    "n15075141": {"label": 999, "name": "toilet_tissue"},
}
# Custom dataset classes for various data sources/formats.
# Each class inherits from torch.utils.data.Dataset and implements __len__ and __getitem__.


class ImageNet(tdata.Dataset):
    """
    PyTorch Dataset for ImageNet.

    Args:
        root (str): Root directory of dataset.
        transform (callable): Transform to apply to images.
        mode (str, optional): Dataset split, e.g., 'train', 'val', 'test'. Default: 'train'.
        classes (list, optional): List of class WordNet IDs to include. Default: all.
        accepted_filetypes (list, optional): List of accepted image file extensions.

    Folder structure:
        root/mode/WordNetID/imagefile.jpg
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.mode = kwargs.get("mode", "train")
        self.classes = kwargs.get("classes", list(LABEL_MAP_IMAGENET.keys()))
        self.accepted_filetypes = kwargs.get("accepted_filetypes", ["png", "jpeg", "jpg"])
        assert isinstance(self.mode, str)
        self.root = root
        self.samples = []

        print("DATA_ROOT", self.root)

        # Build list of (filepath, label) tuples for each class
        for cl in LABEL_MAP_IMAGENET.keys():
            cl_dir = os.path.join(self.root, self.mode, cl)
            if os.path.exists(cl_dir) and cl in self.classes:
                for fname in [f for f in os.listdir(cl_dir) if os.path.isfile(os.path.join(cl_dir, f))]:
                    if fname.split(".")[-1].lower() in self.accepted_filetypes:
                        self.samples.append(
                            (
                                os.path.join(cl_dir, fname),
                                LABEL_MAP_IMAGENET[cl]["label"],
                            )
                        )

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get (image, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed image, label)
        """
        file, label = self.samples[index]
        sample = Image.open(file).convert("RGB")
        sample = self.transform(sample)
        return sample, label


class Food11(tdata.Dataset):
    """
    PyTorch Dataset for Food-11 dataset.

    Args:
        root (str): Root directory of dataset.
        transform (callable): Transform to apply to images.
        mode (str, optional): Dataset split, e.g., 'train', 'val', 'test'. Default: 'train'.
        accepted_filetypes (list, optional): List of accepted image file extensions.

    Folder structure:
        root/mode/readable-class-name/imagefile.jpg
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.mode = kwargs.get("mode", "train")
        self.accepted_filetypes = kwargs.get("accepted_filetypes", ["png", "jpeg", "jpg"])
        assert isinstance(self.mode, str)
        self.root = root

        # Mapping from mode to folder name
        self.modemap = {"train": "training", "test": "evaluation", "val": "validation"}
        self.classes = [
            "Bread",
            "Dairy product",
            "Dessert",
            "Egg",
            "Fried food",
            "Meat",
            "Noodles-Pasta",
            "Rice",
            "Seafood",
            "Soup",
            "Vegetable-Fruit",
        ]
        self.labelmap = {cl: i for i, cl in enumerate(self.classes)}
        self.samples = []

        print("DATA_ROOT", self.root)

        # Build list of (filepath, label) tuples for each class
        for cl in self.labelmap.keys():
            cl_dir = os.path.join(self.root, self.modemap[self.mode], cl)
            if os.path.exists(cl_dir) and cl in self.classes:
                for fname in [f for f in os.listdir(cl_dir) if os.path.isfile(os.path.join(cl_dir, f))]:
                    if fname.split(".")[-1].lower() in self.accepted_filetypes:
                        self.samples.append((os.path.join(cl_dir, fname), self.labelmap[cl]))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get (image, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed image, label)
        """
        file, label = self.samples[index]
        sample = Image.open(file).convert("RGB")
        sample = self.transform(sample)
        return sample, label


class ISIC(tdata.Dataset):
    """
    PyTorch Dataset for ISIC skin lesion dataset.

    Args:
        root (str): Root directory of dataset.
        transform (callable): Transform to apply to images.
        mode (str, optional): Dataset split, 'train' or 'test'. Default: 'train'.
        binary_target (bool, optional): If True, use binary classification (MEL vs. rest).

    Expects metadata CSV file in root directory.
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.root = root
        self.mode = kwargs.get("mode", "train")
        self.binary_target = kwargs.get("binary_target", False)
        super().__init__()
        self.classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

        # Load metadata
        self.metadata = self.construct_metadata()

        # Set class names and class distribution
        if self.binary_target:
            self.class_names = ["Benign", "MEL"]
            num_mel = self.metadata["MEL"].sum()
            dist = np.array([len(self.metadata) - num_mel, num_mel])
        else:
            self.class_names = [
                "MEL",
                "NV",
                "BCC",
                "AK",
                "BKL",
                "DF",
                "VASC",
                "SCC",
                "UNK",
            ]
            dist = np.array([float(x) for x in self.metadata.agg(sum).values[1 : 1 + len(self.class_names)]])

        self.weights = self.compute_weights(dist)

        # Split into train/test indices
        self.idxs_train, self.idxs_test = self.do_train_test_split(0.1)

        # Select metadata for current split
        if self.mode == "train":
            self.idx_metadata = self.metadata.loc[self.idxs_train]
            self.dir = os.path.join(self.root, "Train")
        else:
            self.idx_metadata = self.metadata.loc[self.idxs_test]
            self.dir = os.path.join(
                self.root, "Train"
            )  # Train since we are splitting the train set due to missing test labels
        self.idx_metadata = self.idx_metadata.reset_index(drop=True)

    def do_train_test_split(self, test_split=0.1):
        """
        Stratified train/test split.

        Args:
            test_split (float): Fraction of data to use for test.

        Returns:
            tuple: (train_indices, test_indices)
        """
        rng = np.random.default_rng(0)
        idxs_all = np.arange(len(self.metadata))
        columns = self.metadata.columns.to_list()
        targets = np.array(
            [columns.index(self.metadata.loc[i][self.metadata.loc[i] == 1.0].index[0]) - 1 for i in idxs_all]
        )
        classes = np.unique(targets)
        idxs_classwise = [np.argwhere(targets == cl).squeeze() for cl in classes]
        idxs_test_classwise = [
            np.array(
                sorted(
                    rng.choice(
                        idxs_classwise[cl],
                        size=int(np.round(len(idxs_classwise[cl]) * test_split)),
                        replace=False,
                    )
                )
            )
            for cl in classes
        ]
        idxs_test = np.concatenate(idxs_test_classwise, axis=None)
        idxs_train = np.array(list(set(idxs_all) - set(idxs_test)))
        return idxs_train, idxs_test

    def compute_weights(self, dist):
        """
        Compute class weights for balancing.

        Args:
            dist (np.ndarray): Class distribution.

        Returns:
            torch.Tensor: Class weights.
        """
        return torch.tensor((dist > 0) / (dist + 1e-8) * dist.max()).float()

    def construct_metadata(self):
        """
        Load and process metadata CSV.

        Returns:
            pd.DataFrame: Metadata with unique image IDs.
        """
        data = pd.read_csv(os.path.join(self.root, "ISIC_2019_Training_GroundTruth.csv"))
        data_combined = data.reset_index(drop=True)
        data_combined["isic_id"] = data_combined.image.str.replace("_downsampled", "")
        data_combined = data_combined.drop_duplicates(subset=["isic_id"], keep="last").reset_index(drop=True)
        return data_combined

    def get_all_ids(self):
        """Return list of all image IDs in current split."""
        return list(self.idx_metadata.image.values)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.idx_metadata)

    def __getitem__(self, i):
        """
        Get (image, label) tuple at the specified index.

        Args:
            i (int): Index of the sample.

        Returns:
            tuple: (transformed image, label)
        """
        row = self.idx_metadata.loc[i]
        img = Image.open(os.path.join(self.dir, row["image"] + ".jpg"))
        img = self.transform(img)
        columns = self.idx_metadata.columns.to_list()

        if self.binary_target:
            # 1 = MEL, 0 = non-MEL
            target = (row["MEL"] == 1).astype(int)
        else:
            # Multi-class: find the column with value 1.0 and use its index as label
            target = torch.Tensor([columns.index(row[row == 1.0].index[0]) - 1]).long()[0]
        return img, target


class CUB(tdata.Dataset):
    """
    PyTorch Dataset for Caltech-UCSD Birds (CUB) dataset.

    Args:
        root (str): Root directory of dataset.
        transform (callable): Transform to apply to images.
        mode (str, optional): Dataset split, 'train' or 'test'. Default: 'train'.

    Expects CUB file structure and metadata files.
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.mode = kwargs.get("mode", "train")
        assert isinstance(self.mode, str)
        self.root = root
        self.image_dir = os.path.join(root, "images")
        super().__init__()

        # Load class names
        self.classes = np.loadtxt(self.root + "/classes.txt", dtype=str)
        self.weights = torch.ones(len(self.classes))

        # Load image paths and class labels
        with open(self.root + "/images.txt", "r") as f:
            self.images_paths = [line.split(" ")[-1] for line in f.readlines()]
        with open(self.root + "/image_class_labels.txt", "r") as f:
            self.class_ids = [int(line.split(" ")[-1][:-1]) for line in f.readlines()]
        with open(self.root + "/train_test_split.txt", "r") as f:
            self.train_split = np.array([int(line.split(" ")[-1][:-1]) for line in f.readlines()])

        # Split into train/test
        if self.mode == "train":
            self.class_ids = np.array(self.class_ids)[self.train_split == 0] - 1
            self.images_paths = np.array(self.images_paths)[self.train_split == 0]
        else:
            self.class_ids = np.array(self.class_ids)[self.train_split == 1] - 1
            self.images_paths = np.array(self.images_paths)[self.train_split == 1]

    def __getitem__(self, index):
        """
        Get (image, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed image, label)
        """
        image = self.transform(self.load_image(index))
        return image, self.class_ids[index]

    def load_image(self, index):
        """
        Load image at the specified index.

        Args:
            index (int): Index of the image.

        Returns:
            PIL.Image: Loaded image.
        """
        image_name = ".".join(self.images_paths[index].split(".")[:-1])
        image = Image.open(os.path.join(self.image_dir, f"{image_name}.jpg")).convert("RGB")
        return image

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images_paths)


class SKLearnCircles(tdata.Dataset):
    """
    PyTorch Dataset for sklearn's make_circles synthetic data.

    Args:
        root (str): Path to save/load the generated data (joblib file).
        transform (callable): Transform to apply to samples.
        mode (str, optional): 'train' or 'test'. Default: 'train'.
        redraw (bool, optional): If True, regenerate data even if file exists.

    Generates and caches 2D circle data for toy experiments.
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.num_classes = 2
        self.classes = list(range(self.num_classes))
        self.mode = kwargs.get("mode", "train")
        self.redraw = kwargs.get("redraw", False)
        assert isinstance(self.mode, str)
        self.datafile = root
        self.samples = []

        # Load or generate data
        if not self.redraw and os.path.exists(self.datafile):
            self.samples = joblib.load(self.datafile)
            print("Loaded existing dataset at", self.datafile)
        else:
            n_samples = 10000 if self.mode == "train" else 500
            samples, labels = skdata.make_circles(n_samples, shuffle=False, noise=0.2, factor=0.05)
            for s, lab in list(zip(samples, labels)):
                self.samples.append((s, lab))
            joblib.dump(self.samples, self.datafile)
            print("New Dataset at", self.datafile)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get (sample, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed sample, label)
        """
        sample, label = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def label_to_onehot(self, label):
        """
        Convert label to one-hot encoding.

        Args:
            label (int): Class label.

        Returns:
            np.ndarray: One-hot encoded label.
        """
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot

    def onehot_to_label(self, one_hot):
        """
        Convert one-hot encoding to label(s).

        Args:
            one_hot (np.ndarray): One-hot encoded label.

        Returns:
            list: List of class indices with value 1.
        """
        label = []
        one_hot = one_hot.squeeze()
        for k in range(self.num_classes):
            if one_hot[k] == 1:
                label.append(k)
        return label


class Swirls(tdata.Dataset):
    """
    PyTorch Dataset for synthetic swirl data.

    Args:
        root (str): Path to save/load the generated data (joblib file).
        transform (callable): Transform to apply to samples.
        mode (str, optional): 'train' or 'test'. Default: 'train'.
        redraw (bool, optional): If True, regenerate data even if file exists.

    Generates and caches 2D swirl data for toy experiments.
    """

    def __init__(self, root, transform, **kwargs):
        self.transform = transform
        self.num_classes = 3
        self.classes = list(range(self.num_classes))
        self.mode = kwargs.get("mode", "train")
        self.redraw = kwargs.get("redraw", False)
        assert isinstance(self.mode, str)
        self.datafile = root
        self.samples = []

        # Load or generate data
        if not self.redraw and os.path.exists(self.datafile):
            self.samples = joblib.load(self.datafile)
            print("Loaded existing dataset at", self.datafile)
        else:
            n_samples = 10000 if self.mode == "train" else 500
            samples, labels = self.swirl_the_swirls(n_samples)
            for s, lab in list(zip(samples, labels)):
                self.samples.append((s, lab))
            joblib.dump(self.samples, self.datafile)
            print("New Dataset at", self.datafile)

    def swirl_the_swirls(self, n_samples):
        """
        Generate swirly synthetic data.

        Args:
            n_samples (int): Number of samples to generate.

        Returns:
            tuple: (samples, labels)
        """
        D = 2  # dimensionality
        K = self.num_classes  # number of classes
        N = int(n_samples / K)  # samples per class
        X = np.zeros((N * K, D))  # data matrix (each row = single example)
        y = np.zeros(N * K, dtype="long")  # class labels
        for j in range(K):
            ix = range(N * j, N * (j + 1))
            r = np.linspace(0.0, 1, N)  # radius
            t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j
        return X, y

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get (sample, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed sample, label)
        """
        sample, label = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def label_to_onehot(self, label):
        """
        Convert label to one-hot encoding.

        Args:
            label (int): Class label.

        Returns:
            np.ndarray: One-hot encoded label.
        """
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot

    def onehot_to_label(self, one_hot):
        """
        Convert one-hot encoding to label(s).

        Args:
            one_hot (np.ndarray): One-hot encoded label.

        Returns:
            list: List of class indices with value 1.
        """
        label = []
        one_hot = one_hot.squeeze()
        for k in range(self.num_classes):
            if one_hot[k] == 1:
                label.append(k)
        return label


class SKLearnBlobs(tdata.Dataset):
    """
    PyTorch Dataset for sklearn's make_blobs synthetic data.

    Args:
        root (str): Path to save/load the generated data (joblib file).
        transform (callable): Transform to apply to samples.
        centers (list, optional): List of blob centers.
        mode (str, optional): 'train' or 'test'. Default: 'train'.
        redraw (bool, optional): If True, regenerate data even if file exists.
        n_train_samples (int, optional): Number of training samples.
        n_test_samples (int, optional): Number of test samples.

    Generates and caches 2D blob data for toy experiments.
    """

    def __init__(self, root, transform, centers=[[1, 1], [2, 2]], **kwargs):
        self.transform = transform
        self.num_classes = 2
        self.classes = list(range(self.num_classes))
        self.mode = kwargs.get("mode", "train")
        self.redraw = kwargs.get("redraw", False)
        self.n_train_samples = kwargs.get("n_train_samples", 1000)
        self.n_test_samples = kwargs.get("n_test_samples", 100)
        assert isinstance(self.mode, str)
        self.datafile = root
        self.samples = []

        # Load or generate data
        if not self.redraw and os.path.exists(self.datafile):
            self.samples = joblib.load(self.datafile)
            print("Loaded existing dataset at", self.datafile)
        else:
            n_samples = self.n_train_samples if self.mode == "train" else self.n_test_samples
            samples, labels = skdata.make_blobs(n_samples, centers=centers, cluster_std=0.2, shuffle=False)
            for s, lab in list(zip(samples, labels)):
                self.samples.append((s, lab))
            joblib.dump(self.samples, self.datafile)
            print("New Dataset at", self.datafile)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Get (sample, label) tuple at the specified index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (transformed sample, label)
        """
        sample, label = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def label_to_onehot(self, label):
        """
        Convert label to one-hot encoding.

        Args:
            label (int): Class label.

        Returns:
            np.ndarray: One-hot encoded label.
        """
        one_hot = np.zeros(self.num_classes)
        one_hot[label] = 1
        return one_hot

    def onehot_to_label(self, one_hot):
        """
        Convert one-hot encoding to label(s).

        Args:
            one_hot (np.ndarray): One-hot encoded label.

        Returns:
            list: List of class indices with value 1.
        """
        label = []
        one_hot = one_hot.squeeze()
        for k in range(self.num_classes):
            if one_hot[k] == 1:
                label.append(k)
        return label
