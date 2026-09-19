"""Check native chase compatibility from ordinary observations and static edges.

No simulator objects, hidden predator energy/rest/terrain, or target IDs enter
this module. A compatible move is evidence of possibility, not proof of intent.
"""
import math

from models.entrapment.guide_pathfinding import fixed_frame, RoutePlanner

POSITION_TOLERANCE = 0.75
HEADING_TOLERANCE = 0.05
MISMATCH_TICKS = 2
TERRAIN_MULTIPLIERS = (1., .8, .5, .3)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def possible_follow_moves(sample, geometry):
    """Mirror Predator.step + the native energy/terrain/collision movement gates.

    The previous sample's guide pose is intentional: native DTOs are captured
    just BEFORE that predator moves, so its later displacement targets that pose.
    Each result is (position, heading, reason). Stationary/rest is possible too.
    """
    p,g = sample['predator'],sample['guide']
    heading,guide_heading = sample['predator_heading'],sample['guide_heading']
    distance = math.dist(p,g)
    angle = wrap(math.atan2(g[1]-p[1],g[0]-p[0])-heading)
    looking = wrap(math.atan2(p[1]-g[1],p[0]-g[0])-guide_heading)
    possibilities = [(p,heading,'stationary_or_resting')]

    # Outside hearing range the predator must see the guide to choose it.
    # Skip occlusion filtering conservatively; exact vision-polygon edge cases
    # should not turn plausible pursuit into a confident negative.
    if distance > 60.+1e-7 and (distance > 250.+1e-7 or abs(angle)>math.pi/6+1e-7):
        return possibilities

    commands=[]
    if abs(looking)>math.pi/2 or distance<90.:
        turn=max(-.3,min(.3,angle*.5)) if abs(angle)>.05 else 0.
        direction=turn if abs(angle)>.05 else angle
        commands.append((min(15.,distance),direction,turn,'direct_chase'))
    else:
        # Exact zero gaze is numerically delicate between numpy and math.
        # All three native sign outcomes are possible within roundoff of zero.
        signs=(-1.,0.,1.) if abs(looking)<1e-7 else (-1. if looking>0 else 1.,)
        for sign in signs:
            direction=angle+sign*math.pi/4
            # The source predicts its pivot using 15 units even if energy or
            # terrain later reduces the actual translation.
            x=distance*math.cos(angle)-15.*math.cos(direction)
            y=distance*math.sin(angle)-15.*math.sin(direction)
            commands.append((15.,direction,math.atan2(y,x),'watched_pivot'))

    for requested,direction,turn,reason in commands:
        for cap in (11.,15.):  # Unknown low-energy cap / full sprint.
            for modifier in TERRAIN_MULTIPLIERS:
                length=min(requested,cap)*modifier
                absolute=heading+direction
                q=(p[0]+length*math.cos(absolute),p[1]+length*math.sin(absolute))
                if not geometry.free(q):
                    q=p
                    # Native collision resolution tests 0, -10, +10, -20, ...
                    # degrees, then keeps the original point if all are blocked.
                    for i in range(36):
                        adjusted=absolute+math.pi/18*((i+1)//2)*(-1)**i
                        candidate=(p[0]+length*math.cos(adjusted),p[1]+length*math.sin(adjusted))
                        if geometry.free(candidate):
                            q=candidate
                            break
                possibilities.append((q,wrap(heading+turn),reason))
    return possibilities


def predator_is_not_following(predator, bait, edges, agent, context, memory):
    """Latch initially False; only positive/negative evidence changes the state.

    Unknown, absent, first, ambiguous and resting observations retain the state.
    Actual compatible translation establishes following; repeated incompatible
    movement establishes not-following. Neither is proof of a hidden target ID.
    """
    result=memory.setdefault('_predator_not_following',False)
    tick=context.get('tick',memory.get('_following_call_tick',-1)+1)
    memory['_following_call_tick']=tick
    if predator is None:
        memory.pop('_following_sample',None)
        memory['_following_mismatches']=0
        memory['_following_debug']=dict(status='not_observed',not_following=result)
        return result
    if 'rel_dir' not in predator or sum(o['type']=='Predator' for o in agent['observations'])!=1:
        memory.pop('_following_sample',None)
        memory['_following_mismatches']=0
        memory['_following_debug']=dict(status='ambiguous_identity_or_missing_heading',not_following=result)
        return result

    to_fixed,_=fixed_frame(bait,edges)
    g=to_fixed((0.,0.))
    p=to_fixed((predator['distance']*math.cos(predator['angle']),
                predator['distance']*math.sin(predator['angle'])))
    forward=to_fixed((1.,0.))
    sample=dict(tick=tick,guide=g,predator=p,
                guide_heading=math.atan2(forward[1]-g[1],forward[0]-g[0]),
                predator_heading=wrap(math.atan2(g[1]-p[1],g[0]-p[0])-predator['rel_dir']))
    previous=memory.get('_following_sample')
    memory['_following_sample']=sample
    # At tick 0 the fixture supplies a fresh initial observation. Tick 1 sees
    # that SAME predator pose, not the result of its first movement.
    if previous is None or tick!=previous['tick']+1 or previous['tick']==0:
        memory['_following_mismatches']=0
        memory['_following_debug']=dict(status='need_consecutive_observations',not_following=result)
        return result
    if math.dist(p,previous['predator'])>15.+POSITION_TOLERANCE:
        memory['_following_mismatches']=0
        memory['_following_debug']=dict(status='possible_predator_switch',not_following=result)
        return result
    if '_following_geometry' not in memory:
        memory['_following_geometry']=RoutePlanner([[to_fixed(a),to_fixed(b)] for a,b in edges],clearance=10.)
    candidates=possible_follow_moves(previous,memory['_following_geometry'])
    scored=[(math.dist(p,q),abs(wrap(sample['predator_heading']-h)),reason)
            for q,h,reason in candidates]
    matched=[row for row in scored if row[0]<=POSITION_TOLERANCE and row[1]<=HEADING_TOLERANCE]
    count=0 if matched else memory.get('_following_mismatches',0)+1
    memory['_following_mismatches']=count
    active_match=(math.dist(p,previous['predator'])>POSITION_TOLERANCE
                  and any(row[2]!='stationary_or_resting' for row in matched))
    if active_match:
        result=False
    elif count>=MISMATCH_TICKS:
        result=True
    memory['_predator_not_following']=result
    nearest=min(scored,key=lambda row:row[0]/POSITION_TOLERANCE+row[1]/HEADING_TOLERANCE)
    memory['_following_debug']=dict(
        status=(min(matched,key=lambda row:row[0]+row[1])[2] if matched else 'incompatible_motion'),
        not_following=result,consecutive_mismatches=count,candidate_count=len(candidates),
        closest_position_error=round(nearest[0],4),closest_heading_error=round(nearest[1],4))
    return result
