import sys, math
sys.path.insert(0, "/Users/ekho/code/AI consciousness implementation/cohort")
from verdict import run, InputError

ROSTER = [f"i{i}_o{j}" for i in range(6) for j in range(2)]  # 12 GOLD
MAN = {"tier": "GOLD", "designated_trajectory_ids": ROSTER}

def arm(fn): return {t: fn(t) for t in ROSTER}
def strong(t): return {"A2":0.90,"A3":0.85,"allcut3":0.06}
def chance(t): return {"A2":0.10,"A3":0.08,"allcut3":0.06}
def repl(t): return {"A2":0.90,"A3":0.84,"allcut3":0.06}
def nplus(t): return {"A2":0.80,"A3":0.75,"allcut3":0.06}
def tagok(t): return {"intact":0.95,"cf_track":0.95,"n_total":800}

def base(**over):
    arms = {"R+":arm(strong),"G+":arm(chance),"R-":arm(repl),"G-":arm(chance),"N+":arm(nplus)}
    arms.update(over)
    return {"tier":"GOLD","arms":arms,"gplus_tag_audit":arm(tagok)}

fails=[]
def ck(name, cond):
    print(("PASS" if cond else "FAIL")+": "+name);
    if not cond: fails.append(name)
def raises(name, fn):
    try: fn(); ck(name, False)
    except InputError: ck(name, True)
    except Exception as e: print("FAIL(wrong exc):",name,e); fails.append(name)

r = run(MAN, base())
ck("clean C1 pass", r["C1_BEHAVIORAL"]["pass"] is True)
ck("primary role-equalized pass", r["PRIMARY_ROLE_EQUALIZED"]["pass"] is True)
ck("C3 practical redundancy", r["C3_PRACTICAL_REDUNDANCY"]["status"]=="PRACTICAL_NULL")

# extra-key substitution: drop one designated, add an undesignated -> INPUT_ERROR
def sub():
    a = base(); a["arms"]["R+"] = {t:strong(t) for t in ROSTER if t!="i5_o1"}; a["arms"]["R+"]["EXTRA"]=strong("x")
    run(MAN, a)
raises("extra-key substitution rejected", sub)

# missing designated (no extra) -> INCOMPLETE, denominator not reduced
def miss():
    a = base(); a["arms"]["R+"] = {t:strong(t) for t in ROSTER if t!="i5_o1"}
    return run(MAN, a)
ck("missing run -> C1 INCOMPLETE", miss()["directional"]["C1"]["status"]=="INCOMPLETE")
ck("INCOMPLETE not False", miss()["C1_BEHAVIORAL"]["pass"] is None)

# NaN in one advantaged run -> INPUT_ERROR (not silent pass)
def nan():
    a = base(); a["arms"]["R+"]["i5_o1"]={"A2":float("nan"),"A3":float("nan"),"allcut3":float("nan")}
    run(MAN, a)
raises("NaN advantaged run rejected", nan)

# n+1 records -> INPUT_ERROR
def nplus1():
    a = base(); a["arms"]["R+"]["i6_o0"]=strong("x"); run(MAN,a)
raises("n+1 records rejected", nplus1)

# 0.70 floor exact boundary: adv median d3 exactly 0.70 passes; 0.699 fails
def floor(v):
    a = base(); a["arms"]["R+"]=arm(lambda t:{"A2":0.90,"A3":v,"allcut3":0.06}); return run(MAN,a)
ck("d3 median 0.70 passes floor", floor(0.70)["directional"]["C1"]["components"]["adv_median_d3_ge_0.70"] is True)
ck("d3 median 0.699 fails floor", floor(0.699)["directional"]["C1"]["components"]["adv_median_d3_ge_0.70"] is False)

# tag audit missing denominator -> INPUT_ERROR
def tagnodenom():
    a=base(); a["gplus_tag_audit"]["i0_o0"]={"intact":0.95,"cf_track":0.95}; run(MAN,a)
raises("tag audit without n_total rejected", tagnodenom)

# tag fails -> behavioral pass stays, role-equalized fails (tri-state combine)
def tagbad():
    a=base(); a["gplus_tag_audit"]=arm(lambda t:{"intact":0.4,"cf_track":0.4,"n_total":800}); return run(MAN,a)
ck("behavioral independent of tag", tagbad()["C1_BEHAVIORAL"]["pass"] is True)
ck("role-equalized FAIL when tags unfollowed", tagbad()["PRIMARY_ROLE_EQUALIZED"]["status"]=="FAIL")

# roster size mismatch (5-family manifest) -> InputError
raises("wrong roster size rejected", lambda: run({"tier":"GOLD","designated_trajectory_ids":ROSTER[:10]}, base()))

# reverse-direction counted
rev = run(MAN, base(**{"R+":arm(chance),"G+":arm(strong)}))
ck("reverse direction counted", rev["directional"]["C1"]["components"]["reverse_direction_count"]>=11)

# MINIMUM tier
mman={"tier":"MINIMUM_DECISIVE","designated_trajectory_ids":[f"i{i}_o{j}" for i in range(5) for j in range(2)]}
mr=run(mman,{"tier":"MINIMUM_DECISIVE","arms":{"R+":{t:strong(t) for t in mman["designated_trajectory_ids"]},
     "G+":{t:chance(t) for t in mman["designated_trajectory_ids"]}},
     "gplus_tag_audit":{t:tagok(t) for t in mman["designated_trajectory_ids"]}})
ck("MINIMUM C1 only", list(mr["directional"])==["C1"])
ck("MINIMUM no C3", "C3_PRACTICAL_REDUNDANCY" not in mr)

print("\nALL HARDENED VERDICT TESTS PASS" if not fails else f"\nFAILURES: {fails}")
