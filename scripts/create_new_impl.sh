#!/bin/bash

new_name=$1
caps=${new_name^^}
pattern0="s/SimulationNBodyNaive/SimulationNBody$new_name/g"
pattern1="s/SIMULATION_N_BODY_NAIVE_HPP_/SIMULATION_N_BODY_$caps\_HPP_/g"
cat ./src/murb/implem/SimulationNBodyNaive.cpp   | sed $pattern0 | sed $pattern1  >./src/murb/implem/SimulationNBody$new_name.cpp
cat ./src/murb/implem/SimulationNBodyNaive.hpp   | sed $pattern0 | sed $pattern1  >./src/murb/implem/SimulationNBody$new_name.hpp