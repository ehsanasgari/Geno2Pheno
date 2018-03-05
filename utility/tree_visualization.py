__author__ = "Ehsaneddin Asgari"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Ehsaneddin Asgari"
__email__ = "asgari@berkeley.edu ehsaneddin.asgari@helmholtz-hzi.de"


import random
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace, CircleFace, TextFace, RectFace, random_color, ProfileFace
import matplotlib.pyplot as plt
import random
import matplotlib.patches as mpatches



class VisualizeCircularTree(object):
    '''
        Developed by Ehsaneddin Asgari
    '''
    def __init__(self, nwk_format):
        self.nwk=nwk_format


    def creatCircle(self, filename, name2class_dic, class2color_dic, title, vector=None):
        fg_color_map=['#e5bd89' , '#f8e076' , '#da4f38' , '#ffde3b' ,'#990f02' , '#fe7d68' ,'#66042d' , '#301131' ,'#281e5d' , '#042c36' , '#52b2c0' , '#74b62e' , '#32612d' ,'#522915' , '#2d1606' , '#7e7d9c' , '#f59bbf' , '#ffc30b' , '#fffada' , '#01381b' , '#e96210' , '#6c2d7e' , '#050100' , '#5d2c04' , '#6a6880']
        bg_color_map=['#d5ba9c' , '#b79266' , '#9b7b55' , '#c97f80' , '#b75556' , '#ae4041' , '#fe664e' , '#cc4a34' , '#cf1d01' , '#ff9934' , '#e57100' , '#ff7f00' , '#d8d828' , '#ffff01' , '#ffff01' , '#355f3b' , '#4b6e50' , '#738f76' , '#aeadcd' , '#a4a3cd' , '#7b7a9a' , '#b28e98' , '#e5b6c6' , '#ffcada']
        random.shuffle(bg_color_map)
        random.shuffle(fg_color_map)

        plt.clf()
        axis_font = {'size':'3'}
        plt.rc('xtick', labelsize=0.1)
        plt.rc('ytick', labelsize=0.1)
        plt.rc({'font.size':0.1})


        colors=dict()
        leg=[]
        for idx, value in enumerate(list(set(list(class2color_dic.values())))):
            if value=='unknown' or value=='other':
                colors[value]='white'
            elif idx>= len(bg_color_map):
                colors[value]=VisualizeCircularTree.gen_hex_colour_code()
            else:
                colors[value]=bg_color_map[idx]
            leg.append(mpatches.Patch(color=colors[value], label=value))

        t = Tree(self.nwk)
        # iterate over tree leaves only
        for l in t.iter_leaves():
            ns = NodeStyle()
            ns["bgcolor"] = class2color_dic[name2class_dic[l.name]] if l.name in name2class_dic else 'white'
            l.img_style = ns
            F=TextFace(l.name)
            F.ftype='Times'
            if vector:
                if l.name in vector:
                    l.add_features(profile = vector[l.name])
                    l.add_features(deviation = [0 for x in range(len(vector[l.name]))])
                    l.add_face(ProfileFace(max_v=1, min_v=0.0, center_v=0.5, width=200, height=40, style='heatmap', colorscheme=5), column=0, position="aligned")
        # Create an empty TreeStyle
        ts = TreeStyle()

        # Set our custom layout function
        ts.layout_fn = VisualizeCircularTree.layout

        # Draw a tree
        ts.mode = "c"

        # We will add node names manually
        ts.show_leaf_name = False
        # Show branch data
        ts.show_branch_length = True
        ts.show_branch_support = True
        ts.title.add_face(TextFace(title, fsize=20, ftype='Times'), column=15)

        for k , (value, col) in enumerate(colors.items()):
            x=RectFace(8,8, 'black', col)
            #x.opacity=0.5
            ts.legend.add_face(x, column=8)
            ts.legend.add_face(TextFace(' '+value+'   ', fsize=9,ftype='Times'), column=9)

        t.render(filename+'.pdf',tree_style=ts,dpi=5000)


    @staticmethod
    def gen_hex_colour_code():
        return '#'+''.join([random.choice('0123456789ABCDEF') for x in range(6)])

    @staticmethod
    def layout(node):
        if node.is_leaf():
            # Add node name to laef nodes
            N = AttrFace("name", fsize=14, fgcolor="black")
            faces.add_face_to_node(N, node, 0)
        if "weight" in node.features:
            # Creates a sphere face whose size is proportional to node's
            # feature "weight"
            C = CircleFace(radius=node.weight, color="RoyalBlue", style="sphere")
            # Let's make the sphere transparent
            C.opacity = 0.3
            # And place as a float face over the tree
            faces.add_face_to_node(C, node, 0, position="float")

