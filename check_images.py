#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# TODO 0: Add your information below for Programmer & Date Created.
# PROGRAMMER:
# DATE CREATED:
# REVISED DATE:
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task.
#          Note that the true identity of the pet (or object) in the image is
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results


# Main program function defined below
def main():
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()

    # TODO 1: Define get_input_args function within the file get_input_args.py
    # This function retrieves 3 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args()

    # Function that checks command line arguments using in_arg
    check_command_line_arguments(in_arg)

    # TODO 2: Define get_pet_labels function within the file get_pet_labels.py
    # Once the get_pet_labels function has been defined replace 'None'
    # in the function call with in_arg.dir  Once you have done the replacements
    # your function call should look like this:
    #             get_pet_labels(in_arg.dir)
    # This function creates the results dictionary that contains the results,
    # this dictionary is returned from the function call as the variable results
    results = get_pet_labels(in_arg.dir)
    
    # Function that checks Pet Images in the results Dictionary using results
    check_creating_pet_image_labels(results)

    # TODO 3: Define classify_images function within the file classiy_images.py
    # Once the classify_images function has been defined replace first 'None'
    # in the function call with in_arg.dir and replace the last 'None' in the
    # function call with in_arg.arch  Once you have done the replacements your
    # function call should look like this:
    #             classify_images(in_arg.dir, results, in_arg.arch)
    # Creates Classifier Labels with classifier function, Compares Labels,
    # and adds these results to the results dictionary - results
    classify_images(in_arg.dir, results, in_arg.arch)
    # results = {'cat_01.jpg': ['cat', 'lynx', 0], 'Poodle_07927.jpg': ['poodle', 'standard poodle, poodle', 1], 'cat_02.jpg': ['cat', 'tabby, tabby cat, cat', 1], 'Great_dane_05320.jpg': ['great dane', 'great dane', 1], 'Dalmatian_04068.jpg': ['dalmatian', 'dalmatian, coach dog, carriage dog', 1], 'gecko_02.jpg': ['gecko', 'banded gecko, gecko', 1], 'cat_07.jpg': ['cat', 'egyptian cat, cat', 1], 'Great_pyrenees_05435.jpg': ['great pyrenees', 'great pyrenees', 1], 'German_shepherd_dog_04931.jpg': ['german shepherd dog', 'german shepherd, german shepherd dog, german police dog, alsatian', 1], 'German_shepherd_dog_04890.jpg': ['german shepherd dog', 'german shepherd, german shepherd dog, german police dog, alsatian', 1], 'Collie_03797.jpg': ['collie', 'collie', 1], 'Saint_bernard_08010.jpg': ['saint bernard', 'saint bernard, st bernard', 1], 'Dalmatian_04037.jpg': ['dalmatian', 'dalmatian, coach dog, carriage dog', 1], 'Rabbit_002.jpg': ['rabbit', 'wood rabbit, cottontail, cottontail rabbit, rabbit', 1], 'polar_bear_04.jpg': ['polar bear', 'ice bear, polar bear, ursus maritimus, thalarctos maritimus', 1], 'Poodle_07956.jpg': ['poodle', 'standard poodle, poodle', 1], 'fox_squirrel_01.jpg': ['fox squirrel', 'fox squirrel, eastern fox squirrel, sciurus niger', 1], 'Beagle_01170.jpg': ['beagle', 'walker hound, walker foxhound', 0], 'Boston_terrier_02285.jpg': ['boston terrier', 'boston bull, boston terrier', 1], 'skunk_029.jpg': ['skunk', 'skunk, polecat, wood pussy', 1], 'Boston_terrier_02303.jpg': ['boston terrier', 'boston bull, boston terrier', 1], 'Miniature_schnauzer_06884.jpg': ['miniature schnauzer', 'miniature schnauzer', 1], 'Beagle_01141.jpg': ['beagle', 'beagle', 1], 'Basenji_00974.jpg': ['basenji', 'basenji', 1], 'gecko_80.jpg': ['gecko', 'tailed frog, bell toad, ribbed toad, tailed toad, ascaphus trui', 0], 'Dalmatian_04017.jpg': ['dalmatian', 'dalmatian, coach dog, carriage dog', 1], 'Boxer_02426.jpg': ['boxer', 'boxer', 1], 'Basenji_00963.jpg': ['basenji', 'basenji', 1], 'Boston_terrier_02259.jpg': ['boston terrier', 'boston bull, boston terrier', 1], 'Golden_retriever_05182.jpg': ['golden retriever', 'golden retriever', 1], 'Golden_retriever_05223.jpg': ['golden retriever', 'golden retriever', 1], 'great_horned_owl_02.jpg': ['great horned owl', 'ruffed grouse, partridge, bonasa umbellus', 0], 'Saint_bernard_08036.jpg': ['saint bernard', 'saint bernard, st bernard', 1], 'Golden_retriever_05195.jpg': ['golden retriever', 'golden retriever', 1], 'Beagle_01125.jpg': ['beagle', 'beagle', 1], 'Great_pyrenees_05367.jpg': ['great pyrenees', 'kuvasz', 0], 'German_shorthaired_pointer_04986.jpg': ['german shorthaired pointer', 'german shorthaired pointer', 1], 'Cocker_spaniel_03750.jpg': ['cocker spaniel', 'cocker spaniel, english cocker spaniel, cocker', 1], 'Golden_retriever_05257.jpg': ['golden retriever', 'golden retriever', 1], 'Basset_hound_01034.jpg': ['basset hound', 'basset, basset hound', 1]}
    # Function that checks Results Dictionary using results
    check_classifying_images(results)

    # TODO 4: Define adjust_results4_isadog function within the file adjust_results4_isadog.py
    # Once the adjust_results4_isadog function has been defined replace 'None'
    # in the function call with in_arg.dogfile  Once you have done the
    # replacements your function call should look like this:
    #          adjust_results4_isadog(results, in_arg.dogfile)
    # Adjusts the results dictionary to determine if classifier correctly
    # classified images as 'a dog' or 'not a dog'. This demonstrates if
    # model can correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(results, in_arg.dogfile)

    # Function that checks Results Dictionary for is-a-dog adjustment using results
    check_classifying_labels_as_dogs(results)

    # TODO 5: Define calculates_results_stats function within the file calculates_results_stats.py
    # This function creates the results statistics dictionary that contains a
    # summary of the results statistics (this includes counts & percentages). This
    # dictionary is returned from the function call as the variable results_stats
    # Calculates results of run and puts statistics in the Results Statistics
    # Dictionary - called results_stats
    results_stats = calculates_results_stats(results)

    # Function that checks Results Statistics Dictionary using results_stats
    check_calculating_results(results, results_stats)

    # TODO 6: Define print_results function within the file print_results.py
    # Once the print_results function has been defined replace 'None'
    # in the function call with in_arg.arch  Once you have done the
    # replacements your function call should look like this:
    #      print_results(results, results_stats, in_arg.arch, True, True)
    # Prints summary results, incorrect classifications of dogs (if requested)
    # and incorrectly classified breeds (if requested)
    print_results(results, results_stats, in_arg.arch, True, True)

    # TODO 0: Measure total program runtime by collecting end time
    end_time = time()

    # TODO 0: Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time

    print(
        "\n** Total Elapsed Runtime:",
        str(int((tot_time / 3600))) + ":" + str(int(
            (tot_time % 3600) / 60)) + ":" + str(int((tot_time % 3600) % 60)))


# Call to main function to run the program
if __name__ == "__main__":
    main()
