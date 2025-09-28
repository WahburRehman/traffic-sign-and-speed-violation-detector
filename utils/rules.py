import re

# Traffic sign classes to simple explanation
SIMPLE_RULES = {
    # overtaking / restriction signs
    "no_overtaking": "No overtaking",
    "no_overtaking_trucks": "No overtaking (trucks)",
    "end_no_overtaking": "End of no-overtaking zone",
    "end_no_overtaking_trucks": "End of no-overtaking (trucks)",
    "restriction_ends_80": "End of 80 km/h restriction",  # special case

    # priority / right-of-way
    "priority_road": "Priority road",
    "priority_next_intersection": "Priority at next intersection",
    "give_way": "Give way",
    "stop": "STOP — full stop required",
    "no_entry": "No entry (do not enter)",
    "no_traffic_both_ways": "No traffic in both directions",
    "no_trucks": "No trucks allowed",

    # warnings
    "danger": "General danger",
    "bend_left": "Left bend ahead",
    "bend_right": "Right bend ahead",
    "double_bend": "Double bend ahead",
    "uneven_road": "Uneven road",
    "slippery_road": "Slippery road — drive carefully",
    "road_narrows": "Road narrows ahead",
    "construction": "Road works / Construction ahead",
    "traffic_signal": "Traffic signals ahead",
    "pedestrian_crossing": "Pedestrian crossing",
    "school_crossing": "School crossing ahead",
    "cycles_crossing": "Cycles crossing ahead",
    "snow": "Snow / Ice danger — drive carefully",
    "animals": "Animals crossing ahead",

    # end / restriction lifted
    "restriction_ends": "End of all restrictions",

    # mandatory directions
    "go_right": "Go right only",
    "go_left": "Go left only",
    "go_straight": "Go straight only",
    "go_straight_or_right": "Go straight or right",
    "go_straight_or_left": "Go straight or left",
    "keep_right": "Keep right",
    "keep_left": "Keep left",
    "roundabout": "Roundabout ahead / Enter roundabout",

    # (speed_limit_x handled by regex separately)
}

speed_limit = re.compile(r"speed_limit_(\d+)")

def eval_rule(det_name: str, ctx: dict):
    """
    Returns tuple or None:
      (status, text, priority) where
        status can be {'ok','bad'}  -> banner color
        text -> message to show
        priority: higher wins when multiple signs visible
    ctx may include: {'user_speed': int}
    """
    # 1) Speed-limit with violation logic
    m = speed_limit.match(det_name or "")
    if m:
        limit = int(m.group(1))
        user_speed = int(ctx.get("user_speed", 0))
        violation = user_speed > limit
        text = f"Speed: {user_speed} km/h | Limit: {limit} km/h" + (" | VIOLATION!" if violation else " | OK")
        status = "bad" if violation else "ok"
        return (status, text, 2)

    # 2) Simple informational rules for other signs
    if det_name in SIMPLE_RULES:
        return ("ok", SIMPLE_RULES[det_name], 1)

    # 3) No rule -> no banner
    return None
