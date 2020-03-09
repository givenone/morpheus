OUT_DIR="./out"
TARGET_OBJECT="./models/FACETS_OBJ.obj"
#ANIM_FRAMES_OPTION="--render-anim"

# Make this "true" when testing the scripts
TEST=false
if ${TEST}; then
  RESOLUTION=10
fi

# Create the output directory
mkdir -p ${OUT_DIR}

# Run the scripts
blender --background --python ./run.py --render-frame 1 -- ${OUT_DIR}/01_cube_ ${RESOLUTION} 