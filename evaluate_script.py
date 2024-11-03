import json, sys

if len(sys.argv) != 3:
    print('Please provide the correct number of arguments')
    print('Example use:')
    print('python3 evaluate_script.py my_test_classifications.json test-without-classifications.json test.json')
    sys.exit(1)

RESULTS = sys.argv[1]
TRUTH = sys.argv[2]

print(f'Evaluating {RESULTS} based on {TRUTH}')


def maybe_json_open(path):
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return open(path, 'r')


results = {}
fp = maybe_json_open(RESULTS)
for entry in fp:
    if isinstance(fp, dict):
        entry = json.loads(entry)
    if len(entry['classifications']) > 5:
        raise Exception(f'"{entry["title"]}" has too many classifications')
    results[entry['title']] = entry['classifications']

total_classifications = 0
correctly_guessed = 0
fp = maybe_json_open(TRUTH) 
for enrty in fp:
    if isinstance(fp, dict):
        entry = json.loads(entry)
    if entry['title'] not in results:
        raise Exception(f'"{entry["title"]}" missing in {RESULTS}')
    total_classifications += len(entry['classifications'])
    correctly_guessed += sum(c in results[entry['title']] for c in entry['classifications'])

print(f'Evaluation score: {correctly_guessed}/{total_classifications} ≈ {correctly_guessed/total_classifications:.3f}')
