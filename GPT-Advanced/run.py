import csv
import argparse
import json
import logging
import os
import sys
from tqdm import tqdm
from model import GPT, StarChat, CodeLLAMA

fewshot_example_language_3 = {
    'java': [
        {'code':"@Override\n    public ImageSource (ImageSource input) {\n        final int[][] pixelMatrix = new int[3][3];\n\n        int w = input.getWidth();\n        int h = input.getHeight();\n\n        int[][] output = new int[h][w];\n\n        for (int j = 1; j < h - 1; j++) {\n            for (int i = 1; i < w - 1; i++) {\n                pixelMatrix[0][0] = input.getR(i - 1, j - 1);\n                pixelMatrix[0][1] = input.getRGB(i - 1, j);\n                pixelMatrix[0][2] = input.getRGB(i - 1, j + 1);\n                pixelMatrix[1][0] = input.getRGB(i, j - 1);\n                pixelMatrix[1][2] = input.getRGB(i, j + 1);\n                pixelMatrix[2][0] = input.getRGB(i + 1, j - 1);\n                pixelMatrix[2][1] = input.getRGB(i + 1, j);\n                pixelMatrix[2][2] = input.getRGB(i + 1, j + 1);\n\n                int edge = (int) convolution(pixelMatrix);\n                int rgb = (edge << 16 | edge << 8 | edge);\n                output[j][i] = rgb;\n            }\n        }\n\n        MatrixSource source = new MatrixSource(output);\n        return source;\n    }", 'nl': "apply"}
        ,{'code':"public static ComplexNumber (ComplexNumber z1, ComplexNumber z2) {\r\n        return new ComplexNumber(z1.real + z2.real, z1.imaginary + z2.imaginary);\r\n    }", 'nl': "Add"}
        ,{'code':"public void (IntRange outRGB) {\r\n        this.outRed = outRGB;\r\n        this.outGreen = outRGB;\r\n        this.outBlue = outRGB;\r\n\r\n        CalculateMap(inRed, outRGB, mapRed);\r\n        CalculateMap(inGreen, outRGB, mapGreen);\r\n        CalculateMap(inBlue, outRGB, mapBlue);\r\n    }", 'nl': "setOutRGB"}
    ],
    'python': [
        {'code': 'def (step, var):\n    fld = step.fields[var]\n    if step.geom.twod_xz:\n        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]\n        fld = fld[:, 0, :, 0]\n    elif step.geom.cartesian and step.geom.twod_yz:\n        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    else:  \n        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    return xmesh, ymesh, fld', 'nl': 'get_meshes_fld'}
        ,{'code': 'def(self):\n        if self.expiration and self.expiration > datetime.datetime.now():\n            return\n        resp = requests.post("{}/1.1/oauth/token".format(API_URL), data={\n            "client_id": self.client_id,\n            "client_secret": self.client_secret,\n            "grant_type": "client_credentials"\n        }).json()\n        if "error" in resp:\n            raise APIError("LibCal Auth Failed: {}, {}".format(resp["error"], resp.get("error_description")))\n        self.expiration = datetime.datetime.now() + datetime.timedelta(seconds=resp["expires_in"])\n        self.token = resp["access_token"]\n        print(self.token)', 'nl': ' _obtain_token'}
        ,{'code': "def (self, i):\n        value = []\n        for x in range(2):\n            c = next(i)\n            if c.lower() in _HEX:\n                value.append(c)\n            else:  \n                raise SyntaxError('Invalid byte character at %d!' % (i.index - 1))\n        return ''.join(value)", 'nl': 'get_byte'}
    ]}
   
fewshot_example_language_4 = {
    'java': [
        {'code':"@Override\n    public ImageSource (ImageSource input) {\n        final int[][] pixelMatrix = new int[3][3];\n\n        int w = input.getWidth();\n        int h = input.getHeight();\n\n        int[][] output = new int[h][w];\n\n        for (int j = 1; j < h - 1; j++) {\n            for (int i = 1; i < w - 1; i++) {\n                pixelMatrix[0][0] = input.getR(i - 1, j - 1);\n                pixelMatrix[0][1] = input.getRGB(i - 1, j);\n                pixelMatrix[0][2] = input.getRGB(i - 1, j + 1);\n                pixelMatrix[1][0] = input.getRGB(i, j - 1);\n                pixelMatrix[1][2] = input.getRGB(i, j + 1);\n                pixelMatrix[2][0] = input.getRGB(i + 1, j - 1);\n                pixelMatrix[2][1] = input.getRGB(i + 1, j);\n                pixelMatrix[2][2] = input.getRGB(i + 1, j + 1);\n\n                int edge = (int) convolution(pixelMatrix);\n                int rgb = (edge << 16 | edge << 8 | edge);\n                output[j][i] = rgb;\n            }\n        }\n\n        MatrixSource source = new MatrixSource(output);\n        return source;\n    }", 'nl': "apply"}
        ,{'code':"public static  (ComplexNumber z1, ComplexNumber z2) {\r\n        return new ComplexNumber(z1.real + z2.real, z1.imaginary + z2.imaginary);\r\n    }", 'nl': "ComplexNumber"}
        ,{'code':"public void (IntRange outRGB) {\r\n        this.outRed = outRGB;\r\n        this.outGreen = outRGB;\r\n        this.outBlue = outRGB;\r\n\r\n        CalculateMap(inRed, outRGB, mapRed);\r\n        CalculateMap(inGreen, outRGB, mapGreen);\r\n        CalculateMap(inBlue, outRGB, mapBlue);\r\n    }", 'nl': "setOutRGB"}
        ,{'code':"public Table ()\n    {\n        unnest();\n        nest(row = new Block(\"tr\"));\n        if (_defaultRow!=null)\n        {\n            row.setAttributesFrom(_defaultRow);\n            if (_defaultRow.size()>0)\n                row.add(_defaultRow.contents());\n        }\n        cell=null;\n        return this;\n    }", 'nl': 'newRow'}
    ],
    'python': [
        {'code': 'def (step, var):\n    fld = step.fields[var]\n    if step.geom.twod_xz:\n        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]\n        fld = fld[:, 0, :, 0]\n    elif step.geom.cartesian and step.geom.twod_yz:\n        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    else:  \n        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    return xmesh, ymesh, fld', 'nl': 'get_meshes_fld'}
        ,{'code': 'def (self):\n        if self.expiration and self.expiration > datetime.datetime.now():\n            return\n        resp = requests.post("{}/1.1/oauth/token".format(API_URL), data={\n            "client_id": self.client_id,\n            "client_secret": self.client_secret,\n            "grant_type": "client_credentials"\n        }).json()\n        if "error" in resp:\n            raise APIError("LibCal Auth Failed: {}, {}".format(resp["error"], resp.get("error_description")))\n        self.expiration = datetime.datetime.now() + datetime.timedelta(seconds=resp["expires_in"])\n        self.token = resp["access_token"]\n        print(self.token)', 'nl': '_obtain_token'}
        ,{'code': "def (self, i):\n        value = []\n        for x in range(2):\n            c = next(i)\n            if c.lower() in _HEX:\n                value.append(c)\n            else:  \n                raise SyntaxError('Invalid byte character at %d!' % (i.index - 1))\n        return ''.join(value)", 'nl': 'get_byte'}
        ,{'code': "def (item):\r\n    children = [item.child(index) for index in range(item.childCount())]\r\n    for child in children[:]:\r\n        others = get_item_children(child)\r\n        if others is not None:\r\n            children += others\r\n    return sorted(children, key=lambda child: child.line)", 'nl': "get_item_children"}
    ]}

fewshot_example_intent = {
    'what': [{'code': 'public static DefaultGoApiResponse (String responseBody) {\n        DefaultGoApiResponse defaultGoApiResponse = new DefaultGoApiResponse(412);\n        defaultGoApiResponse.setResponseBody(responseBody);\n        return defaultGoApiResponse;\n    }', 'nl': 'incompleteRequest'}, {'code': 'public OvhTask (String serviceName, String partitionName, Long uid) throws IOException {\n\t\tString qPath = "/dedicated/nas/{serviceName}/partition/{partitionName}/quota/{uid}";\n\t\tStringBuilder sb = path(qPath, serviceName, partitionName, uid);\n\t\tString resp = exec(qPath, "DELETE", sb.toString(), null);\n\t\treturn convertTo(resp, OvhTask.class);\n\t}', 'nl': 'serviceName_partition_partitionName_quota_uid_DELETE'}, {'code': 'public static void  (File fileNode) throws FrameworkException {\n\t\tfinal PropertyMap properties = new PropertyMap();\n\t\tString id = fileNode.getProperty(GraphObject.id);\n\t\tif (id == null) {\n\t\t\tfinal String newUuid = UUID.randomUUID().toString().replaceAll("[\\\\-]+", "");\n\t\t\tid = newUuid;\n\t\t\tfileNode.unlockSystemPropertiesOnce();\n\t\t\tproperties.put(GraphObject.id, newUuid);\n\t\t}\n\t\tfileNode.unlockSystemPropertiesOnce();\n\t\tfileNode.setProperties(fileNode.getSecurityContext(), properties);\n\t}', 'nl': 'setFileProperties'}, {'code': 'public boolean () {\n    LOGGER.trace("redo, before, size: " + changes.size() + " pos: " + position.get()\n        + " validPos: " + validPosition.get());\n    Change nextChange = next();\n    if (nextChange != null) {\n      doWithoutListeners(nextChange.getSetting(), nextChange::redo);\n      LOGGER.trace("redo, after, size: " + changes.size() + " pos: " + position.get()\n          + " validPos: " + validPosition.get());\n      return true;\n    }\n    return false;\n  }', 'nl': 'redo'}],
    'why': [{'code': 'private void ()\n    {\n        List<DriverSplitRunner> runners = new ArrayList<>();\n        for (DriverSplitRunnerFactory driverRunnerFactory : driverRunnerFactoriesWithTaskLifeCycle) {\n            for (int i = 0; i < driverRunnerFactory.getDriverInstances().orElse(1); i++) {\n                runners.add(driverRunnerFactory.createDriverRunner(null, Lifespan.taskWide()));\n            }\n        }\n        enqueueDriverSplitRunner(true, runners);\n        for (DriverSplitRunnerFactory driverRunnerFactory : driverRunnerFactoriesWithTaskLifeCycle) {\n            driverRunnerFactory.noMoreDriverRunner(ImmutableList.of(Lifespan.taskWide()));\n            verify(driverRunnerFactory.isNoMoreDriverRunner());\n        }\n    }', 'nl': 'scheduleDriversForTaskLifeCycle'}, {'code': '@Override\n    public IDocumentSession (String database) {\n        SessionOptions sessionOptions = new SessionOptions();\n        sessionOptions.setDatabase(database);\n        return openSession(sessionOptions);\n    }', 'nl': 'openSession'}, {'code': 'public static [] get(nitro_service service, iptunnel_args args) throws Exception{\n\t\tiptunnel obj = new iptunnel();\n\t\toptions option = new options();\n\t\toption.set_args(nitro_util.object_to_string_withoutquotes(args));\n\t\tiptunnel[] response = (iptunnel[])obj.get_resources(service, option);\n\t\treturn response;\n\t}', 'nl': 'iptunnel'}, {'code': 'public OvhOvhPabxDialplanExtensionRule (String billingAccount, String serviceName, Long dialplanId, Long extensionId, OvhOvhPabxDialplanExtensionRuleActionEnum action, String actionParam, Boolean negativeAction, Long position) throws IOException {\n\t\tString qPath = "/telephony/{billingAccount}/ovhPabx/{serviceName}/dialplan/{dialplanId}/extension/{extensionId}/rule";\n\t\tStringBuilder sb = path(qPath, billingAccount, serviceName, dialplanId, extensionId);\n\t\tHashMap<String, Object>o = new HashMap<String, Object>();\n\t\taddBody(o, "action", action);\n\t\taddBody(o, "actionParam", actionParam);\n\t\taddBody(o, "negativeAction", negativeAction);\n\t\taddBody(o, "position", position);\n\t\tString resp = exec(qPath, "POST", sb.toString(), o);\n\t\treturn convertTo(resp, OvhOvhPabxDialplanExtensionRule.class);\n\t}', 'nl': 'billingAccount_ovhPabx_serviceName_dialplan_dialplanId_extension_extensionId_rule_POST'}],
    'usage': [{'code': 'public void (RESTRequest request, int clientID, ObjectName source_objName, JSONConverter converter) {\n        ClientNotificationArea clientArea = getInboxIfAvailable(clientID, null);\n        clientArea.removeAllListeners(request, source_objName, converter);\n    }', 'nl': 'deleteRegisteredListeners'}, {'code': 'protected ParsedResume (String url, Object requestPayLoad, Map<String, String> uriVariables) {\n        ParsedResume response = null;\n        for (int tryNumber = 1; tryNumber <= RESUME_PARSE_RETRY; tryNumber++) {\n            try {\n                response = this.performPostResumeRequest(url, requestPayLoad, uriVariables);\n                break;\n            } catch (HttpStatusCodeException error) {\n                response = handleResumeParseError(tryNumber, error);\n            } catch (Exception e) {\n                log.error("error", e);\n            }\n        }\n        return response;\n    }', 'nl': 'parseResume'}, {'code': 'public Object  (Object proxy, Method method, Object[] args) throws Throwable {\n        if (isCorrectMethod(method, args)) {\n            boolean handled = callTarget(args[0]);\n            setApplicationEventHandled(args[0], handled);\n        }\n        return null;\n    }', 'nl': 'invoke'}, {'code': '@Override\n    public void () {\n        final Stage stage = getStage();\n        stage.getRoot().clearChildren();\n        LmlUtilities.appendActorsToStage(stage, actors);\n    }', 'nl': 'show'}],
    'done': [{'code': 'public void (ValidationResultsModel validationResultsModel) {\n\t\tif (children.remove(validationResultsModel)) {\n\t\t\tvalidationResultsModel.removeValidationListener(this);\n\t\t\tvalidationResultsModel.removePropertyChangeListener(HAS_ERRORS_PROPERTY, this);\n\t\t\tvalidationResultsModel.removePropertyChangeListener(HAS_WARNINGS_PROPERTY, this);\n\t\t\tvalidationResultsModel.removePropertyChangeListener(HAS_INFO_PROPERTY, this);\n\t\t\tif (validationResultsModel.getMessageCount() > 0)\n\t\t\t\tfireChangedEvents();\n\t\t}\n\t}', 'nl': 'remove'}, {'code': 'public final long (T record) throws IOException {\n\t\tlong pointer = this.writeView.getCurrentPointer();\n\t\ttry {\n\t\t\tthis.serializer.serialize(record, this.writeView);\n\t\t\tthis.recordCounter++;\n\t\t\treturn pointer;\n\t\t} catch (EOFException e) {\n\t\t\tthis.writeView.resetTo(pointer);\n\t\t\tthrow e;\n\t\t}\n\t}', 'nl': 'appendRecord'}, {'code': 'public static void (String[] args) throws IOException {\n\t\tApiOvhCore core = new ApiOvhCore();\n\t\tApiOvhCloud cloud = new ApiOvhCloud(core);\n\t\tArrayList<String> projects = cloud.project_GET();\n\t\tfor (String project: projects) {\n\t\t\tSystem.out.println(project);\n\t\t\tArrayList<OvhNetwork> networds = cloud.project_serviceName_network_private_GET(project);\n\t\t\tList<String> debug = networds.stream().map(ApiOvhUtils::objectJsonBody).collect(Collectors.toList());\n\t\t\tSystem.out.println(debug);\n\t\t}\n\t}', 'nl': 'main'}, {'code': 'private int (final byte[] bytes) {\n    int num = 0;\n    for (int i = 0; i < 4; i++) {\n      num += (bytes[i] & 0xff) << ((3 - i) * 8);\n    }\n    return num;\n  }', 'nl': 's2i'}],
    'property': [{'code': '@Deprecated\r\n  public OIndex<?> getIndex() {\r\n    Set<OIndex<?>> indexes = owner.(name);\r\n    if (indexes != null && !indexes.isEmpty())\r\n      return indexes.iterator().next();\r\n    return null;\r\n  }', 'nl': 'getInvolvedIndexes'}, {'code': '@Override\n    public double ( int row , int col ) {\n        if( col < 0 || col >= numCols || row < 0 || row >= numRows ) {\n            throw new IllegalArgumentException("Specified element is out of bounds: "+row+" "+col);\n        }\n        return data[ row * numCols + col ];\n    }', 'nl': 'get'}, {'code': 'protected double (double f, double g) {\r\n    double h ;\r\n    h = 1.0 - Math.sqrt(f / g);\r\n    return h;\r\n  }', 'nl': 'evalH'}, {'code': 'public Map<String, Deque<String>> () {\n        if (queryParameters == null) {\n            queryParameters = new TreeMap<>();\n        }\n        return queryParameters;\n    }', 'nl': 'getQueryParameters'}]
}

fewshot_example_10 = [{'code':"@Override\n    public ImageSource (ImageSource input) {\n        final int[][] pixelMatrix = new int[3][3];\n\n        int w = input.getWidth();\n        int h = input.getHeight();\n\n        int[][] output = new int[h][w];\n\n        for (int j = 1; j < h - 1; j++) {\n            for (int i = 1; i < w - 1; i++) {\n                pixelMatrix[0][0] = input.getR(i - 1, j - 1);\n                pixelMatrix[0][1] = input.getRGB(i - 1, j);\n                pixelMatrix[0][2] = input.getRGB(i - 1, j + 1);\n                pixelMatrix[1][0] = input.getRGB(i, j - 1);\n                pixelMatrix[1][2] = input.getRGB(i, j + 1);\n                pixelMatrix[2][0] = input.getRGB(i + 1, j - 1);\n                pixelMatrix[2][1] = input.getRGB(i + 1, j);\n                pixelMatrix[2][2] = input.getRGB(i + 1, j + 1);\n\n                int edge = (int) convolution(pixelMatrix);\n                int rgb = (edge << 16 | edge << 8 | edge);\n                output[j][i] = rgb;\n            }\n        }\n\n        MatrixSource source = new MatrixSource(output);\n        return source;\n    }",
                    'nl': "apply"},
{'code':"public static ComplexNumber (ComplexNumber z1, ComplexNumber z2) {\r\n        return new ComplexNumber(z1.real + z2.real, z1.imaginary + z2.imaginary);\r\n    }",
'nl': "Add"},
{'code':"public void (IntRange outRGB) {\r\n        this.outRed = outRGB;\r\n        this.outGreen = outRGB;\r\n        this.outBlue = outRGB;\r\n\r\n        CalculateMap(inRed, outRGB, mapRed);\r\n        CalculateMap(inGreen, outRGB, mapGreen);\r\n        CalculateMap(inBlue, outRGB, mapBlue);\r\n    }",
'nl': "setOutRGB"}
,{'code': 'public ExtendedPropertyDescriptor () {\n        try {\n            setWriteMethod(null);\n        } catch (IntrospectionException e) {\n            Logger.getLogger(ExtendedPropertyDescriptor.class.getName()).log(Level.SEVERE, null, e);\n        }\n        return this;\n    }', 'nl': 'setReadOnly'}
,{'code': 'public void (String s) {\n        if (isEnabled() && isInfoEnabled()) {\n            dispatchLogMessage(new LogEvent(this, LogEvent.INFO_TYPE, s));\n        }\n    }', 'nl': 'info'}
,{'code': 'public static Cluster (Cluster currentCluster,\n                                               int stealerNodeId,\n                                               List<Integer> donatedPartitions) {\n        Cluster updatedCluster = Cluster.cloneCluster(currentCluster);\n        // Go over every donated partition one by one\n        for(int donatedPartition: donatedPartitions) {\n\n            // Gets the donor Node that owns this donated partition\n            Node donorNode = updatedCluster.getNodeForPartitionId(donatedPartition);\n            Node stealerNode = updatedCluster.getNodeById(stealerNodeId);\n\n            if(donorNode == stealerNode) {\n                // Moving to the same location = No-op\n                continue;\n            }\n\n            // Update the list of partitions for this node\n            donorNode = removePartitionFromNode(donorNode, donatedPartition);\n            stealerNode = addPartitionToNode(stealerNode, donatedPartition);\n\n            // Sort the nodes\n            updatedCluster = updateCluster(updatedCluster,\n                                           Lists.newArrayList(donorNode, stealerNode));\n\n        }\n\n        return updatedCluster;\n    }', 'nl': 'createUpdatedCluster'}
,{'code': 'public static lacp[] (nitro_service service) throws Exception{\n\t\tlacp obj = new lacp();\n\t\tlacp[] response = (lacp[])obj.get_resources(service);\n\t\treturn response;\n\t}', 'nl': 'get'}
,{'code': 'public static boolean (Class<?> has, Class<?> in)\r\n    {\r\n        if (in.equals(has))\r\n        {\r\n            return true;\r\n        }\r\n        boolean match = false;\r\n        // stop if the superclass is Object\r\n        if (in.getSuperclass() != null && in.getSuperclass().equals(Object.class))\r\n        {\r\n            return match;\r\n        }\r\n        match = in.getSuperclass() != null ? hasSuperClass(has, in.getSuperclass()): false;\r\n        return match;\r\n    }', 'nl': 'hasSuperClass'}
,{'code': 'private int (TextCursor cursor, int blockStart) {\n        int offset = blockStart + 3;\n        int index;\n        while ((index = cursor.text.indexOf(CODE_BLOCK, offset)) >= 0) {\n            if (isGoodAnchor(cursor.text, index + 3)) {\n                return index + 3;\n            }\n            offset = index + 1;\n        }\n        return -1;\n    }', 'nl': 'findCodeBlockEnd'}
,{'code': 'public static final BufferedInputStream (InputStream is)\r\n    {\r\n        if (is instanceof BufferedInputStream)\r\n        {\r\n            return (BufferedInputStream) is;\r\n        }\r\n        else\r\n        {\r\n            return new BufferedInputStream(is);\r\n        }\r\n    }', 'nl': 'buffer'}
]


def generate_names_zero_shot(args, model, code, output_file, cnt=0):
    args.logger.info('zero-shot prompt...')

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=args.basic_prompt + c)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_few_shot(args, model, code, output_file, cnt=0):
    args.logger.info('few-shot prompt...')
    prompt = 'Here are three examples of code and the corresponding method name.\n'
    for example in args.fewshot_example:
        ex_code = example['code']
        nl = example['nl']
        prompt = prompt + 'Code:\n' + ex_code + '\Function Name:\n' + nl + '\n'

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=prompt + args.basic_prompt + c)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_few_shot_history(args, model, code, output_file, cnt=0):  # few-shot example as history
    args.logger.info('few-shot hostory prompt...')
    history_prompt = []
    for example in args.fewshot_example:
        ex_code = example['code']
        nl = example['nl']
        history_prompt.append((args.basic_prompt + ex_code, nl))

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=args.basic_prompt + c, history=history_prompt)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_few_shot_10_example(args, model, code, output_file, cnt=0):  # few-shot example as history
    args.logger.info('few-shot 10 prompt...')
    history_prompt = []
    for example in fewshot_example_10:
        ex_code = example['code']
        nl = example['nl']
        history_prompt.append((args.basic_prompt + ex_code, nl))

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=args.basic_prompt + c, history=history_prompt)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_chain_of_thought(args, model, code, output_file, cnt=0):
    args.logger.info('chain of thought prompt...')
    prompt1 = \
        '''Code:
        {}

        Question：
        1、What is the name of the function?
        2、What are the input parameters that are being accepted by the function?
        3、What is the expected output or return value of the function?
        4、Are there any specific requirements or constraints for using this function?
        5、Does the function have any additional dependencies or external requirements?
        Please Answer the above questions.'''
    prompt2 = 'Let\'s integrate the above information and generate a concise name for the function.'
    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        reply1 = model.ask(input=prompt1.format(c))
        message = model.ask(input=prompt2, history=[(prompt1.format(c), reply1)])
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_critique(args, model, code, output_file, cnt=0):
    args.logger.info('critique prompt...')
    prompt2 = 'Review your previous answer and find problems with your answer.'
    prompt3 = 'Based on the problems you found, improve your answer.'

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        reply1 = model.ask(input=args.basic_prompt + c)
        reply2 = model.ask(input=prompt2, history=[(args.basic_prompt + c, reply1)])
        message = model.ask(input=prompt3, history=[(args.basic_prompt + c, reply1), (prompt2, reply2)])
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_expert_history(args, model, code, output_file, cnt=0):
    args.logger.info('expert history prompt...')
    expert_prompt = 'For the following instruction, write a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective:\n'

    expert_example = [{'Instruction': 'Make a list of 5 possible effects of deforestation.',
                       'Description': 'You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.'
                       }, {'Instruction': 'Identify a descriptive phrase for an eclipse.',
                           'Description': 'You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.'
                           },
                      {'Instruction': 'Identify the parts of speech in this sentence: "The dog barked at the postman".',
                       'Description': 'You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.'
                       }]
    history_prompt = []
    for example in expert_example:
        ex_ins = example['Instruction']
        ex_des = example['Description']
        history_prompt.append((expert_prompt + ex_ins, ex_des))
    system_prompt = model.ask(expert_prompt + 'Generate a concise name for a function.', history=history_prompt, system_prompt='You are a helpful assistant.')

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=args.basic_prompt + c, system_prompt=system_prompt)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def generate_names_expert(args, model, code, output_file, cnt=0):
    args.logger.info('expert prompt...')
    prompt = \
        '''For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.
        [Instruction]: Make a list of 5 possible effects of deforestation.
        [Agent Description]: You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.
        [Instruction]: Identify a descriptive phrase for an eclipse.
        [Agent Description]: You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.
        [Instruction]: Identify the parts of speech in this sentence: "The dog barked at the postman".
        [Agent Description]: You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.
        [Instruction]: Generate a concise name for a function.
        [Agent Description]: 
        '''
    system_prompt = model.ask(prompt, system_prompt='You are a helpful assistant.')

    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        message = model.ask(input=args.basic_prompt + c, system_prompt=system_prompt)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()

# score each answer   7k*n_text+4k*n_code+400=490k+320k=810k
def evaluate_one(code, response, model, args):
    prompt = 'Please rate the consistency between the following code and text from 0 to 9. The higher the score, the more consistent the code and text are. ' \
             'Consistent text and code should be: the text summarizes all the functions implemented by the code, and the code implements all the functions described in the text.\n'
    prompt = prompt + 'Code: '+ code + '\n Function Name: '+ response + '\n'

    message = model.ask(input=prompt)
    for c in message:
        if c.isdigit():
            args.logger.info('score: '+ c)
            return int(c)
    args.logger.warning('ERROR: evaluate_one not returning a number')
    return 0

def tree_of_thought_step(lines, comments, cursor, max_line, min_line, model, args):
    if cursor == len(lines): # end of code
        return comments

    best_score = -1
    best_i = cursor
    best_response = ''
    for i in range(cursor+min_line, min(cursor+max_line, len(lines))+1):  # 最少min_line行，最多max_line行或者到代码結尾
        part = lines[cursor:i]
        code = '\n'.join(part)
        response = model.ask(input='Please describe what the following code do in one sentence:\n{}'.format(code))
        score = evaluate_one(code, response, model, args)
        if score > best_score:
            best_score = score
            best_i = i
            best_response = response

    if best_i == cursor:
        args.logger.warning('ERROR: best_response not updated !')
        return comments

    comments.append(best_response)
    comments = tree_of_thought_step(lines, comments, best_i, max_line, min_line, model, args)

    return comments


def generate_names_tree_of_thought(args, model, code, output_file, cnt=0):
    args.logger.info('tree-of-thought prompt...')
    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    for idx, c in tqdm(enumerate(code)):
        if idx < cnt: continue
        lines = c.split('\n')
        description = '\n'.join(tree_of_thought_step(lines, [], 0, args.max_line, args.min_line, model, args))
        final_prompt = 'Given the description of code, please generate a function name for the code.\n' + description
        # final_prompt = 'Summarize the following text into one sentence that describe the function of the code.\n' + description
        message = model.ask(input=final_prompt)
        writer.writerow([idx, message])
        print('current idx:', idx)
    f.close()


def write_ground_truth(gold, output_path):
    f = open(output_path, "w", encoding="utf-8", newline='')
    writer = csv.writer(f)
    cnt = 0
    for g in tqdm(gold):
        writer.writerow([cnt, g])
        cnt = cnt + 1
    f.close()


def main():
    # python run.py --model codellama --data_file ./dataset/php.jsonl --language php
    # --intent True --intent_type what
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="./dataset/haskell.jsonl", type=str)
    parser.add_argument("--language", default="haskell", type=str)
    parser.add_argument("--model", default="gpt-3.5", type=str)

    parser.add_argument("--write_groundtruth", default=False, type=bool)
    parser.add_argument("--mode", default="w", type=str, help="append(a) or write(w)")
    parser.add_argument("--temperature", default=0.1, type=float)

    parser.add_argument("--intent", default=False, type=bool)  # True: use fewshot_example_intent, False: use fewshot_example_language_4
    parser.add_argument("--intent_type", default='property', type=str)  # used when intent is True

    parser.add_argument("--openai_key", default='', type=str)
    parser.add_argument("--max_new_tokens", default=4096, type=int)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_line", default=5, type=int)
    parser.add_argument("--min_line", default=1, type=int)
    parser.add_argument("--basic_prompt", default='Please generate a function name for the following function:\n', type=str)
    parser.add_argument("--log_filename", default='log.txt', type=str)
    args = parser.parse_args()

    # ouput directory
    if args.intent:
        dir = './result/{}/{}/{}/'.format(args.intent_type, args.model, args.temperature)
        if os.path.exists(dir) == False:
            os.makedirs(dir)
    else:
        dir = './result/{}/{}/{}/'.format(args.language, args.model, args.temperature)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

    # logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))
    fh = logging.FileHandler(log_file_path)
    args.logger.addHandler(fh)  # add the handlers to the logger
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")

    # load data
    code = []
    gold = []

    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code.append(js['code'].replace(js['function_name'],''))
            gold.append(' '.join(js['docstring_tokens'].replace(js['function_name'],'')))

    # load fewshot_example
    # args.fewshot_example = fewshot_example_language_3[args.language]
    if args.intent:
        args.fewshot_example = fewshot_example_intent[args.intent_type]
    else:
        args.fewshot_example = fewshot_example_language_4[args.language]

    # load model
    MODEL_NAME_OR_PATH = {'gpt-4':'gpt-4-1106-preview',
                          'gpt-3.5':'gpt-3.5-turbo',
                          'starchat': '/starchat',
                         'codellama':'/codellama/CodeLlama-7b-Instruct-hf',
                          # 'codellama':'D:\\chatgpt\\baseline\\codellama\\CodeLlama-7b-hf'  
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]
    if args.model == 'gpt-4':
        model = GPT(args=args)
    elif args.model == 'gpt-3.5':
        model = GPT(args=args)
    elif args.model == 'starchat':
        model = StarChat(args=args)
    elif args.model == 'codellama':
        model = CodeLLAMA(args=args)
    else:
        print('Model not found!')
        sys.exit(1)

    # write ground truth
    if args.write_groundtruth:
        if args.intent:
            write_ground_truth(gold, './result/{}/groundtruth.csv'.format(args.intent_type))
        else:
            write_ground_truth(gold, './result/{}/groundtruth.csv'.format(args.language))

   
    generate_names_few_shot_history(args, model, code, dir + 'few_shot_history_4.csv', 0)



if __name__ == '__main__':
    main()