import csv
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from model import GPT, StarChat, CodeLLAMA, StarCoder

from util.remove_comments import remove_comments_and_docstrings
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
tokenizer_name = RobertaTokenizer.from_pretrained("/XXX/XXX/XXX/", local_files_only=True )


fewshot_example_10 = [
    {
        "code": "void (Runnable runnable, ExecuteStepMessage executeStepMessage, int timeout, TimeUnit unit) {\n    if (!isRunningAStep.getAndSet(true)) {\n        this.currentlyExecutingStep = executeStepMessage;\n        Future<String> future = null;\n        try {\n            future = scheduledExecutorService.submit(runStepAndResetIsRunning(runnable), \"OK\");\n            future.get(timeout, unit);\n        } catch (TimeoutException e) {\n            future.cancel(true);\n            log.warn(\"A step failed to execute within \" + timeout + \" \" + unit + \", attempting to cancel the step\");\n        } catch (Exception e) {\n            String ms = \"Exception while executing step [\" + e.getMessage() + \"]\";\n            log.error(ms, e);\n            stepFailureConsumer.accept(ms, executeStepMessage);\n        }\n    } else {\n        String message = \"Cannot execute a test step, a step is already in progress [\" + currentlyExecutingStep.getStepId() + \", \" + currentlyExecutingStep.getPattern() + \"]\";\n        log.error(message);\n        stepFailureConsumer.accept(message, executeStepMessage);\n    }\n}",
        "nl": "runWithinPeriod"
    },
    {
        "code": "private void (String fileOffset, boolean prefixNewLine) throws IOException {\n    long now = System.currentTimeMillis();\n    LogEntry entry = new LogEntry(now, componentID, fileOffset);\n    String line = entry.toString();\n    if (prefixNewLine) {\n        lockFileStream.writeBytes(System.lineSeparator() + line);\n    } else {\n        lockFileStream.writeBytes(line);\n    }\n    lockFileStream.hflush();\n    lastEntry = entry;\n}",
        "nl": "logProgress"
    },
    {
        "code": "public void (final Node child) {\n    if (child instanceof Level) {\n        levels.add((Level) child);\n        nodes.add(child);\n        if (child.getParent() != null) child.removeParent();\n        child.setParent(this);\n    } else if (child instanceof SpecTopic) {\n        appendSpecTopic((SpecTopic) child);\n    } else if (child instanceof CommonContent) {\n        commonContents.add((CommonContent) child);\n        nodes.add(child);\n        if (child.getParent() != null) child.removeParent();\n        child.setParent(this);\n    } else {\n        nodes.add(child);\n        if (child.getParent() != null) child.removeParent();\n        child.setParent(this);\n    }\n}",
        "nl": "appendChild"
    },
    {
        "code": "public HttpClient (final URI uri) {\n    final URIBuilder builder = new URIBuilder(uri);\n    this.scheme = builder.getScheme();\n    this.host = builder.getHost();\n    this.port = builder.getPort();\n    this.path = builder.getPath();\n    this.fragment = builder.getFragment();\n    this.resetQueryParameters();\n    for (NameValuePair nvp : builder.getQueryParams()) {\n        this.queryParameters.add(new BasicNameValuePair(nvp.getName(), nvp.getValue()));\n    }\n    return this;\n}",
        "nl": "setURI"
    },
    {
        "code": "public String (int column) throws SQLException {\n    Field field = getField(column);\n    String result = connection.getTypeInfo().getJavaClass(field.getOID());\n    if (result != null) return result;\n    int sqlType = getSQLType(column);\n    switch (sqlType) {\n        case Types.ARRAY:\n            return \"java.sql.Array\";\n        default:\n            String type = getPGType(column);\n            if (\"unknown\".equals(type)) return \"java.lang.String\";\n            return \"java.lang.Object\";\n    }\n}",
        "nl": "getColumnClassName"
    },
    {
        "code": "public static GroupElement (final Curve curve, final FieldElement ypx, final FieldElement ymx, final FieldElement xy2d) {\n    return new GroupElement(curve, Representation.PRECOMP, ypx, ymx, xy2d, null);\n}",
        "nl": "precomp"
    },
    {
        "code": "public static TransferListener (String xferId, AsperaTransaction transaction) {\n    if (instance == null) instance = new TransferListener();\n    if (transactions.get(xferId) != null) {\n        transactions.get(xferId).add(transaction);\n    } else {\n        List<AsperaTransaction> transferTransactions = new ArrayList<>();\n        transferTransactions.add(transaction);\n        transactions.put(xferId, transferTransactions);\n    }\n    return instance;\n}",
        "nl": "getInstance"
    },
    {
        "code": "private DescribeSecurityGroupsResponseType () {\n    DescribeSecurityGroupsResponseType ret = new DescribeSecurityGroupsResponseType();\n    ret.setRequestId(UUID.randomUUID().toString());\n    SecurityGroupSetType securityGroupSet = new SecurityGroupSetType();\n    for (MockSecurityGroup item : mockSecurityGroupController.describeSecurityGroups()) {\n        SecurityGroupItemType securityGroupItem = new SecurityGroupItemType();\n        securityGroupItem.setOwnerId(MOCK_SECURITY_OWNER_ID);\n        securityGroupItem.setGroupName(item.getGroupName());\n        if (!DEFAULT_MOCK_PLACEMENT.getAvailabilityZone().equals(currentRegion)) {\n            securityGroupItem.setGroupId(currentRegion + \"_\" + item.getGroupId());\n            securityGroupItem.setVpcId(currentRegion + \"_\" + item.getVpcId());\n        } else {\n            securityGroupItem.setGroupId(item.getGroupId());\n            securityGroupItem.setVpcId(item.getVpcId());\n        }\n        securityGroupItem.setGroupDescription(item.getGroupDescription());\n        IpPermissionSetType ipPermissionSet = new IpPermissionSetType();\n        for (MockIpPermissionType mockIpPermissionType : item.getIpPermissions()) {\n            IpPermissionType ipPermission = new IpPermissionType();\n            ipPermission.setFromPort(mockIpPermissionType.getFromPort());\n            ipPermission.setToPort(mockIpPermissionType.getToPort());\n            ipPermission.setIpProtocol(mockIpPermissionType.getIpProtocol());\n            ipPermissionSet.getItem().add(ipPermission);\n        }\n        IpPermissionSetType ipPermissionEgressSet = new IpPermissionSetType();\n        for (MockIpPermissionType mockIpPermissionType : item.getIpPermissionsEgress()) {\n            IpPermissionType ipPermission = new IpPermissionType();\n            ipPermission.setFromPort(mockIpPermissionType.getFromPort());\n            ipPermission.setToPort(mockIpPermissionType.getToPort());\n            ipPermission.setIpProtocol(mockIpPermissionType.getIpProtocol());\n            ipPermissionEgressSet.getItem().add(ipPermission);\n        }\n        securityGroupItem.setIpPermissionsEgress(ipPermissionEgressSet);\n        securityGroupSet.getItem().add(securityGroupItem);\n    }\n    ret.setSecurityGroupInfo(securityGroupSet);\n    return ret;\n}",
        "nl": "describeSecurityGroups"
    },
    {
        "code": "public DynaFormControl (final Object data, final String type, final int colspan, final int rowspan) {\n    DynaFormControl dynaFormControl = new DynaFormControl(data, type, colspan, rowspan, row, elements.size() + 1, dynaFormModel.getControls().size() + 1, extended);\n    elements.add(dynaFormControl);\n    dynaFormModel.getControls().add(dynaFormControl);\n    totalColspan += colspan;\n    return dynaFormControl;\n}",
        "nl": "addControl"
    },
    {
        "code": "public static boolean (Method... methods) {\n    for (Method method : methods) {\n        if (!Modifier.isStatic(method.getModifiers())) {\n            return false;\n        }\n    }\n    return true;\n}",
        "nl": "areAllMethodsStatic"
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
