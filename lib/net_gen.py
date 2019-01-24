""" Utility functions for generating network modules of various sizes from given ".tsv" files """

import os, glob, sys
import subprocess
import toposort
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import lib.graph_util as graph_util

def __generate_nets(gnw_path, main_netfile_path, net_size, num_nets_per_size, outdir_path):
    """ Generate random network modules extracted from main_net_filename
        Args:
            gnw_path: path to GeneNetWeaver ".jar" file.
            main_netfile_path: filepath for the main network used for module extraction.
            net_size: size of the extracted modules.
            num_nets_per_size: number of modules to be extracted for each size.
            outdir_path: directory to store extracted network modules.
    """

    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)
    command = ("java -jar {!s}  --extract -c data/settings.txt --input-net {!s}"
               " --input-net-format=0 --random-seed --greedy-selection --subnet-size={!s}"
               " --num-subnets={!s} --output-net-format=0 --output-path={!s}"
               ).format(gnw_path, main_netfile_path, net_size, num_nets_per_size, outdir_path)
    subprocess.run([command], shell=True)

    file_prefix = os.path.basename(main_netfile_path)[:-4]
    generated_files = glob.glob("{!s}/{!s}*.tsv".format(outdir_path, file_prefix))
    return generated_files

def __generate_dependency_graphs(dir_path):
    """ Generate dependency graph in ".dep" files for each ".tsv" file in directory.
        Args:
            dir_path: directory path containing ".tsv" files.
    """

    graph_filenames =  [x for x in os.listdir(dir_path) if x.endswith(".tsv")]
    for filename in graph_filenames:
        file_path = '{!s}/{!s}'.format(dir_path, filename)
        dep_graph_file_path = '{!s}.dep'.format(file_path[:-4])

        try:
            g = graph_util.DirGraph(file_path)
            g.save_as_linear_dep_graph(dep_graph_file_path)

            print("Generated: {!s}".format(dep_graph_file_path))
        except toposort.CircularDependencyError:
            print("Loop detected, cannot generate {!s}".format(dep_graph_file_path))
        except:
            raise

def run(gnw_path, min_netsize, max_netsize, stepsize, num_nets_per_size, main_netfile_path, output_prefix):
    """ Run net_gen for provided configs 
    """
    all_generated_files = []

    for i in range(min_netsize, max_netsize, stepsize):
        dir_path="{!s}{:d}".format(output_prefix, i)
        curr_generated_files = __generate_nets(gnw_path, main_netfile_path, i, num_nets_per_size, dir_path)
        all_generated_files.extend(curr_generated_files)
        __generate_dependency_graphs(dir_path)
    
    return all_generated_files
