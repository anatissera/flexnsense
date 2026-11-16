# Problems and how they were found

| Problem | How it showed | Cause | Fix |
|---|---|---|---|
| The arms went through the scale | In the video | The scale was only drawn | The tube became a physical body with collisions, hinged to the pull arm's jaws (`08`) |
| Driving the gripper turned the wrist | A user remark, then `sim/dbg_gripper.py` (four arms, one difference each) | In the simple conversion, the gear's collision shape pushed on the forearm | The gear motor is left free; the two jaws are driven directly; no collisions between the arm's own parts |
| The move arm "held" the scale but was not touching it | A user remark ("does it look connected?"), then measuring where the tube was in the gripper's frame | Several things, below | |
| ... the jaws closed on nothing | A close-up render: the gripper had no fingers | In the finer-collision conversion, the three gripper joints (both jaws, the gear) had frames whose two sides disagreed by 36 and 87 degrees, so the physics engine put the jaws about 10 cm off the gripper | The joint frames are recomputed from where the parts are drawn (`09_fix_gripper_usd.py`, and inside `08`) |
| ... the arm reached past the tube | The measured position along the scale | It aimed at the cap and loop, which are only drawn | The grasp point moved onto the tube, 150 mm from the ring |
| ... the scene blew up when the jaws touched the tube | Joint values of metres | A 12 g jaw pinching a 50 g tube is numerically unstable | The jaws get the apparent mass of their gearbox (0.3 kg, assumed); 240 physics steps a second; more solver passes |
| The finger pads were invisible, then enormous | A render | The jaw parts are modelled in millimetres (scale 0.001) | Pads built directly with explicit sizes, and their measured size printed |
| The live stream (WebRTC) did not connect | The user's screen | Probably the venue network blocking the video port | Recordings saved to `videos/` instead |
| Training episodes ended after a quarter of a second | Episode length in the log | The gear motor was driven in the training task | Gear left free there too |
| The move policy cheated twice | Lift height and time to place in the evaluation | See `03-training.md` | Rules and rewards corrected |

What worked as a method: when numbers and expectation disagree, **look at a picture early**. An hour went into
tuning jaw forces before one close-up render showed the fingers were not there.
