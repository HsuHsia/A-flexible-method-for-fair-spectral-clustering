def judge(res1, res2, res):
    r1_ed_inc = 1 - (res1.ed / res.ed)
    r2_ed_inc = 1 - (res2.ed / res.ed)
    r1_b_inc = (res1.balance - res.balance) / res.balance
    r2_b_inc = (res2.balance - res.balance) / res.balance

    r1_inc = (r1_ed_inc + r1_b_inc) / 2
    r2_inc = (r2_ed_inc + r2_b_inc) / 2
    print("inc")
    print(r1_inc)
    print(r2_inc)

    return res1 if r1_inc > r2_inc else res2

