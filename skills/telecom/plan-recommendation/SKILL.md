---
name: telecom-plan-recommendation
description: 根据移联电信用户的近期流量使用、语音通话分钟数、当前套餐费用和有效候选套餐，给出有数据依据的套餐建议。凡用户询问“推荐套餐”“套餐是否合适”“流量经常不够”“通话多该换什么套餐”“想降低月租但怕不够用”等场景，都应使用此 Skill。只负责分析和推荐；单纯查询当前套餐、明确指定套餐的变更、故障排查、漫游办理或电商请求不使用此 Skill。
version: 0.1.0
domain: telecom
capabilities:
  - plan_recommendation
allowed_agent_types:
  - telecom_agent
allowed_tools:
  - telecom_get_current_plan
  - telecom_get_usage_profile
  - telecom_list_plans
  - telecom_compare_plans
effect: read
---

# Telecom Plan Recommendation

## Goal

Recommend a suitable mobile plan from active plans by comparing the user's actual data and voice usage with expected monthly cost and capacity. Prefer a defensible recommendation over an aggressive upgrade.

This Skill is advisory and read-only. A later request to change the plan must enter the separate plan-change write flow and obtain explicit confirmation.

## Required reference

Before comparing plans, load [references/recommendation-policy.md](references/recommendation-policy.md). It defines the usage window, insufficient-data handling, ranking priorities, and response fields.

## Preconditions

- A request-level user context and target line must be available.
- The line must belong to the request user.
- Use only active plans returned by tools. Never invent a plan, price, allowance, discount, or eligibility rule.
- Do not ask the user to provide data already available from the permitted read tools.

If the user has multiple lines and has not identified one, ask which line they want analyzed before retrieving usage.

## Workflow

1. Call `telecom_get_current_plan` for the selected line.
2. Call `telecom_get_usage_profile` to obtain recent completed billing cycles and the current-cycle projection for:
   - mobile data usage in MB;
   - voice call usage in minutes;
   - optional overage/refuel history;
   - data quality and number of usable cycles.
3. Call `telecom_list_plans` to obtain active and line-eligible candidate plans.
4. If the usage profile is insufficient, follow the insufficient-data path in the reference instead of pretending the estimate is reliable.
5. Call `telecom_compare_plans` with the usage profile reference and candidate plan IDs. Treat its price, capacity, expected cost, and eligibility results as authoritative.
6. Rank the eligible results using the reference policy.
7. Recommend staying on the current plan when it already meets usage needs and no alternative has a meaningful expected benefit.
8. Return one primary recommendation and at most two useful alternatives. Exclude dominated plans that are both more expensive and less suitable.

## Recommendation behavior

- Balance expected monthly cost with enough data and voice capacity.
- Avoid recommending a high-priced plan merely because it has the largest allowance.
- Explain whether the recommendation is driven mainly by data, voice calls, overage/refuels, or cost savings.
- Distinguish current known usage from projections.
- State important tradeoffs, such as lower monthly price with less buffer or higher price with more stable capacity.
- If no eligible plan improves the user's situation, recommend keeping the current plan.
- If tools return conflicting or unavailable business data, state that a reliable recommendation cannot currently be produced.

## Write boundary

Do not call a plan-change tool and do not claim that a plan has been changed. If the user accepts a recommendation, hand the selected `plan_id` and the quoted comparison facts to the separate plan-change flow. That flow must re-read the current line and plan state, present the exact impact, create a pending action, and obtain a second confirmation before any database write.

## Output contract

Use this structure in the result supplied to the response node:

```text
analysis_period
usage_summary:
  data_usage
  voice_minutes
  overage_or_refuel_pattern
  data_quality
current_plan:
  plan_id
  name
  monthly_price
primary_recommendation:
  plan_id
  name
  monthly_price
  expected_monthly_cost
  reasons[]
  tradeoffs[]
alternatives[]:
  plan_id
  name
  expected_monthly_cost
  best_for
keep_current_plan: bool
clarification_question: optional
```

The user-facing response should summarize the evidence in plain Chinese and make clear that this is a recommendation, not an executed change.

## Examples

### Should trigger

- “我这几个月流量老是不够，电话也比较多，帮我推荐一个合适套餐。”
- “现在套餐月租有点贵，结合我的使用情况看看有没有更划算的。”
- “我每月大概打很多电话但流量不多，适合什么套餐？”

### Should not trigger

- “查询我当前是什么套餐。” — use the direct current-plan read capability.
- “把我的套餐改成畅享 50G。” — use the plan-change flow; do not recommend unless the user asks for comparison.
- “手机没有信号怎么办？” — use Telecom troubleshooting RAG.
- “帮我退掉商城订单。” — route to Retail.
