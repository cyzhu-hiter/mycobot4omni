# This file is for model version checking.

# Scipts to convert asset instanceable
from omniisaacgymenvs.utils.usd_utils.create_instanceable_assets import convert_asset_instanceable as A
A(
    asset_usd_path=ASSET_USD_PATH, 
    source_prim_path=SOURCE_PRIM_PATH, 
    save_as_path=SAVE_AS_PATH
)

mycobot_v3: May 29th, 2023 -> mycobot_instance: May 30th, 2023
    Driver Parameters Updated.
    Generated based on mycobot_v3
    Bugs:
	1. env_0 initialization, gripper gets twisted;
	2. gear_joint is not working;
	3. tip cannot be applied to RigidPrimView, instead of just  XformPrimView. 

mycobot_v4: June 1st, 2023 -> mycobot_v4_instance.usd
    1. Add distance between the flange and gripper base
    2. Disable collidor of two gear joints
    3. Move tip out of the xform, gripper_base and create a new Xform "_tip" to help retrieving RigidPrimView.

mycobot_v5: June 5th, 2023 -> mycobot_v5_instance.usd
    1. Apply angular joint directly on right gear_joint, instead of using gear joint.
    2. Delete collision set of two hinge parts.

mycobot_v6: June 6th, 2023 -> mycobot_v6_instance.usd
    1. Add a xform called "_camera_loc" at camera position to help locate
    2. Replace two revolute hinge_finger_joint with spherical joint.
