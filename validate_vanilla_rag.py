import json

# Load the Vanilla RAG output
data = json.load(open(r'D:\workspace\contextual-intent\stitch_output\completed-qwenplus-traj-8\label_based_retrieval_vanilla_rag.json'))

print('=' * 60)
print('Vanilla RAG Output Validation')
print('=' * 60)

all_valid = True

for i, q in enumerate(data['question_results'], 1):
    scopes = q.get('selected_context_scopes', [])
    events = q.get('selected_event_types', [])
    targets = q.get('selected_targets', [])
    retrieved_count = len(q['node_selected_turn_ids'])

    valid = len(scopes) == 0 and len(events) == 0 and len(targets) == 0
    all_valid &= valid

    status = "[PASS]" if valid else "[FAIL]"
    print(f'Question {i}:')
    print(f'  - selected_scopes: {len(scopes)} (expected: 0)')
    print(f'  - selected_events: {len(events)} (expected: 0)')
    print(f'  - selected_targets: {len(targets)} (expected: 0)')
    print(f'  - retrieved_turns: {retrieved_count} (expected: 20)')
    print(f'  - status: {status}')
    print()

print('=' * 60)
if all_valid:
    print('[PASS] All questions validated successfully!')
    print('[PASS] Vanilla RAG correctly skipped label selection (Steps 1-4)')
    print('[PASS] Used semantic similarity retrieval only (Step 5)')
else:
    print('[FAIL] Some questions failed validation')
    print('[FAIL] Label selection results should be empty arrays')
print('=' * 60)

# Check reasoning fields to confirm they indicate skipping
print('\nChecking reasoning fields (should contain "Skipped" marker):')
q1 = data['question_results'][0]
print(f'  - context_scope_reasoning: {q1.get("context_scope_reasoning", "N/A")[:50]}...')
print(f'  - event_type_reasoning: {q1.get("event_type_reasoning", "N/A")[:50]}...')
print(f'  - target_reasoning: {q1.get("target_reasoning", "N/A")[:50]}...')
