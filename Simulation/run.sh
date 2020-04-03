OUT_DIR="./out"
TARGET_OBJECT="./models/FACETS_OBJ.obj"

#ANIM_FRAMES_OPTION="--render-anim"

# Make this "true" when testing the scripts
TEST=false
if ${TEST}; then
  RESOLUTION=10
fi

# Export path
#export PATH=$PATH:$1

# Create the output directory
#mkdir -p ${OUT_DIR}

# Run the scripts
blender --background --python ./run.py --render-frame 1 -- "C:\\Users\\user\\Desktop\\lightstage\\morpheus\\Simulation\\output\\cycle_test_revised_8_hdr" "HDR"
blender --background --python ./run.py --render-frame 1 -- "C:\\Users\\user\\Desktop\\lightstage\\morpheus\\Simulation\\output\\cycle_test_revised_9_png" "PNG"