import csv
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from model import GPT, StarChat, CodeLLAMA，StarCoder

from util.remove_comments import remove_comments_and_docstrings
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True )

fewshot_example_10 = [
    {
        "code": "def ( self , param , author = None , constraints = None , origin = '' ) : if isinstance ( param , SkillEntry ) : skill = param else : skill = self . find_skill ( param , author ) entry = build_skill_entry ( skill . name , origin , skill . is_beta ) try : skill . install ( constraints ) entry [ 'installed' ] = time . time ( ) entry [ 'installation' ] = 'installed' entry [ 'status' ] = 'active' entry [ 'beta' ] = skill . is_beta except AlreadyInstalled : entry = None raise except MsmException as e : entry [ 'installation' ] = 'failed' entry [ 'status' ] = 'error' entry [ 'failure_message' ] = repr ( e ) raise finally : if entry : self . skills_data [ 'skills' ] . append ( entry )",
        "nl": "install"
    },
    {
        "code": "def ( * lazy_properties ) : def wrap_class ( cls ) : if not hasattr ( cls , 'load' ) : raise ValueError ( \"no load method implemented\" ) for lazy_property in lazy_properties : setattr ( cls , lazy_property , property ( * lazy_property_factory ( lazy_property ) ) ) if not hasattr ( cls , '__getstate__' ) : def __getstate__ ( self ) : serializable_state = self . __dict__ . copy ( ) for lazy_property in lazy_properties : attr = serializable_state . get ( '_' + lazy_property ) if isinstance ( attr , collections . Iterator ) : raise ValueError ( \"Iterators can't be lazy loaded\" ) serializable_state . pop ( '_' + lazy_property , None ) return serializable_state setattr ( cls , '__getstate__' , __getstate__ ) return cls return wrap_class",
        "nl": "do_not_pickle_attributes"
    },
    {
        "code": "def ( self , expectation , discard_result_format_kwargs = True , discard_include_configs_kwargs = True , discard_catch_exceptions_kwargs = True ) : new_expectation = copy . deepcopy ( expectation ) if \"success_on_last_run\" in new_expectation : del new_expectation [ \"success_on_last_run\" ] if discard_result_format_kwargs : if \"result_format\" in new_expectation [ \"kwargs\" ] : del new_expectation [ \"kwargs\" ] [ \"result_format\" ] if discard_include_configs_kwargs : if \"include_configs\" in new_expectation [ \"kwargs\" ] : del new_expectation [ \"kwargs\" ] [ \"include_configs\" ] if discard_catch_exceptions_kwargs : if \"catch_exceptions\" in new_expectation [ \"kwargs\" ] : del new_expectation [ \"kwargs\" ] [ \"catch_exceptions\" ] return new_expectation",
        "nl": "copy_and_clean_up_expectation"
    },
    {
        "code": "def ( ncVar ) : attributes = ncVarAttributes ( ncVar ) if not attributes : return None for key in ( 'missing_value' , 'MissingValue' , 'missingValue' , 'FillValue' , '_FillValue' ) : if key in attributes : missingDataValue = attributes [ key ] return missingDataValue return None",
        "nl": "variable_missing_value"
    },
    {
        "code": "def ( self , bin ) : bin_path = ( self . _bin_dir / bin ) . with_suffix ( \".exe\" if self . _is_windows else \"\" ) if not bin_path . exists ( ) : return bin return str ( bin_path )",
        "nl": "bin"
    },
    {
        "code": "def ( self , name , default , help , constant = False ) : self . AddOption ( type_info . List ( name = name , default = default , description = help , validator = type_info . Integer ( ) ) , constant = constant )",
        "nl": "DEFINE_integer_list"
    },
    {
        "code": "def ( self ) : def first_run ( ) : self . _ioloop_thread_id = get_thread_ident ( ) if self . OBSERVE_UPDATES : self . attach ( ) self . ioloop . add_callback ( first_run )",
        "nl": "start"
    },
    {
        "code": "def ( reraise_exceptions = False , * * kwargs ) : exit_status = 0 try : cli . main ( * * kwargs ) except SoftLayer . SoftLayerAPIError as ex : if 'invalid api token' in ex . faultString . lower ( ) : print ( \"Authentication Failed: To update your credentials, use 'slcli config setup'\" ) exit_status = 1 else : print ( str ( ex ) ) exit_status = 1 except SoftLayer . SoftLayerError as ex : print ( str ( ex ) ) exit_status = 1 except exceptions . CLIAbort as ex : print ( str ( ex . message ) ) exit_status = ex . code except Exception : if reraise_exceptions : raise import traceback print ( \"An unexpected error has occured:\" ) print ( str ( traceback . format_exc ( ) ) ) print ( \"Feel free to report this error as it is likely a bug:\" ) print ( \"    https://github.com/softlayer/softlayer-python/issues\" ) print ( \"The following snippet should be able to reproduce the error\" ) exit_status = 1 sys . exit ( exit_status )",
        "nl": "main"
    },
    {
        "code": "def ( self , params ) : schedule = self . server . schedule n = float ( params . get ( 'n' ) ) e = float ( params . get ( 'e' ) ) s = float ( params . get ( 's' ) ) w = float ( params . get ( 'w' ) ) limit = int ( params . get ( 'limit' ) ) stops = schedule . GetStopsInBoundingBox ( north = n , east = e , south = s , west = w , n = limit ) return [ StopToTuple ( s ) for s in stops ]",
        "nl": "handle_json_GET_boundboxstops"
    },
    {
        "code": "def ( self ) : self . funcs . update ( { 'system.listMethods' : self . system_listMethods , 'system.methodSignature' : self . system_methodSignature , 'system.methodHelp' : self . system_methodHelp } )",
        "nl": "register_introspection_functions"
    }
]

def generate_summaries_zero_shot(args, model, code, output_file, cnt=0):
    args.logger.info('zero-shot prompt...')

    with open(output_file, args.mode, encoding="utf-8") as f:
        for idx, c in tqdm(enumerate(code)):
            if idx < cnt:
                continue

            message = model.ask(input=args.basic_prompt + c)

            json_line = {
                "idx": idx,
                "code": c,
                "prediction": message
            }
            json.dump(json_line, f, ensure_ascii=False)
            f.write('\n')

            print('current idx:', idx)


def generate_summaries_few_shot(args, model, code, output_file, cnt=0):
    args.logger.info('few-shot prompt...')
    prompt = 'Here are ten examples of code and the corresponding method name.\n'
    for example in args.fewshot_example_10:
        ex_code = example['code']
        nl = example['nl']
        prompt = prompt + 'Code:\n' + ex_code + '\nfunc_name:\n' + nl + '\n'

    with open(output_file, args.mode, encoding="utf-8") as f:
        for idx, c in tqdm(enumerate(code)):
            if idx < cnt:
                continue

            message = model.ask(input=args.basic_prompt + c)

            json_line = {
                "idx": idx,
                "code": c,
                "prediction": message
            }
            json.dump(json_line, f, ensure_ascii=False)
            f.write('\n')

            print('current idx:', idx)


def write_ground_truth(gold, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, g in tqdm(enumerate(gold)):
            json_line = {"idx": idx, "gold": g}
            json.dump(json_line, f, ensure_ascii=False)
            f.write('\n')

def replace_first_func_name(raw_code, func_name):
    idx = raw_code.find(func_name)
    if idx == -1:
        return raw_code
    return raw_code[:idx] + " " + raw_code[idx + len(func_name):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="/XXX/XXX/XXX.jsonl", type=str)
    parser.add_argument("--language", default="XXX", type=str)
    parser.add_argument("--model", default="XXX", type=str)

    parser.add_argument("--write_groundtruth", default=True, type=bool)
    parser.add_argument("--mode", default="w", type=str, help="append(a) or write(w)")
    parser.add_argument("--temperature", default=0.01, type=float)

    parser.add_argument("--openai_key", default='XXX', type=str)
    parser.add_argument("--max_new_tokens", default=0, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_line", default=5, type=int)
    parser.add_argument("--min_line", default=1, type=int)
    parser.add_argument("--basic_prompt", default='Please generate the appropriate method name based on the code snippet:\n', type=str)
    parser.add_argument("--log_filename", default='log.txt', type=str)
    args = parser.parse_args()


    dir = './result/{}/{}/{}/'.format(args.language, args.model, args.temperature)
    if os.path.exists(dir) == False:
        os.makedirs(dir)


    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))
    fh = logging.FileHandler(log_file_path)
    args.logger.addHandler(fh)
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")


    codes = []
    gold = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):

            line = line.strip()
            if not line:
                continue

            js = json.loads(line)


            func_name = js['func_name'].replace('\n', ' ').split('.')[-1].strip()


            raw_code = js["code"]
            code_without_name = replace_first_func_name(raw_code, func_name)

            try:
                clean_code = remove_comments_and_docstrings(source=code_without_name, lang="java")
            except Exception as e:
                print(f"[Warning] Failed to remove comments for idx {idx}: {e}")
                clean_code = code_without_name

            codes.append(clean_code)


            gold_name = func_name.lower().rstrip('0123456789')
            gold_tokens = tokenizer_name.tokenize(gold_name)
            gold_str = ' '.join(gold_tokens).replace('\n', '')


            gold_str = ' '.join(gold_str.strip().split())


            result = ''.join(gold_str.split())
            gold.append(result)

    MODEL_NAME_OR_PATH = {'gpt-3.5':'gpt-3.5-turbo',
                          'codellama': '/XXX/XXX/XXX/codellama-I',
                          'starchat': '/XXX/XXX/XXX/starchat-bert',
			  'starcoder': '//XXX/XXX/XXX/starcoderbase-7b'
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
    if args.model == 'gpt-3.5':
        model = GPT(args=args)
    elif args.model == 'starchat':
        model = StarChat(args=args)
    elif args.model == 'codellama':
        model = CodeLLAMA(args=args)
    elif args.model == 'starcoder':
        model = StarCoder(args=args)
    else:
        print('Model not found!')
        sys.exit(1)


    if args.write_groundtruth:
        write_ground_truth(gold,  dir + '/groundtruth.jsonl')


    generate_summaries_zero_shot(args, model, codes, dir + 'zero_shot.jsonl', 0)
    # generate_summaries_few_shot_10_example(args, model, code, dir + 'few_shot_10_example.csv', 0)

if __name__ == '__main__':
    main()
