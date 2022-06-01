import argparse
import subprocess

parser = argparse.ArgumentParser(description='Multi-Object-Grasping')
parser.add_argument('-l', '--obj_list', help='delimited list input', type=str)
args = parser.parse_args()
obj_list = [str(item) for item in args.obj_list.split(',')]

num_objs = len(obj_list)
xml_path = 'mog_xmls/object_xmls/'

# Add first line
with open(xml_path+"n_o_objects.xml", "w") as f1:
    f1.write('<mujocoinclude>')
    f1.write('\n')

for obj_ind in range(num_objs):
    # Add object xml
    with open(xml_path+obj_list[obj_ind]+".xml") as f:
        lines = f.readlines()
        lines = [l for l in lines]
        with open(xml_path+"n_o_objects.xml", "a") as f1:
            f1.writelines(lines)

# Add last line
with open(xml_path+"n_o_objects.xml", "a") as f1:
    f1.write('</mujocoinclude>')
