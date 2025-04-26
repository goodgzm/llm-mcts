import json
import random
import sys
from typing import Dict, Any, Optional


import math
import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import os
import torch.nn as nn
from torch.distributions.categorical import Categorical
import copy

# root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(root)
#
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


class Node(object):
    """
        basic tree node
    """
    def __init__(
        self, parent: "Node" = None,
    ) -> None:
        self._parent = parent
        self._children = {}
        # 标记节点是否终止
        self._terminated = False

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the leaf node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

class LanguageNode(Node):
    """
        LLM tree node
    """
    # 当前节点环境观测状态文本描述
    text_state: Optional[str] = None
    # 当前节点执行的最后一个动作（智能体上一步执行的动作）
    last_action: Optional[str] = None
    # 已生成的 token 数量
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        state: Optional[str] = None,
        num_generated_token: Optional[int] = None,
        task=1,
        last_action=None,
        history_prompt="",
    ) -> None:
        super().__init__(parent)
        # 当前节点环境观测状态，可以写 state 或 state, 因为 basic MCTS 的 state 并不会有多个
        self.state = state
        self.task = task
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.last_action = last_action    
        self.history_prompt = history_prompt                                            

    def get_path(self):
        paths = []
        node = self
        while not node.is_root():
            paths.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(paths))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
        else:
            info_dict["text_state"] = self.text_state
        return info_dict


    def obs2text(self, obs):

        text = ""

        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]

        see_pancake = obs[4]
        close_to_pancake = obs[5]
        hold_pancake = obs[6]

        see_microwave = obs[7]
        close_to_microwave = obs[8]
        is_microwave_open = obs[9]

        pancake_in_microwave = obs[10]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        object_text = ""
        action_list = []

        if in_kitchen:

            if not see_pancake:
                object_text += "The pancake is in the microwave. "
            else:
                object_text += "You notice pancake and microwave. "

            if hold_pancake:
                object_text += "Currently, you have grabbed the pancake in hand. "
                if close_to_microwave:
                    object_text += "The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 7, 8, 9]
                else:
                    object_text += "The microwave is not close to you. "
                    action_list = [0, 2, 3, 4, 5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is close to you. "
                    action_list = [0, 2, 3, 5, 6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 8, 9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not close to you. "
                    action_list = [0, 2, 3, 4, 5]
                else:
                    if is_microwave_open:
                        action_list = [0, 2, 3, 8, 9]
                    else:
                        action_list = [0, 2, 3, 9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0, 2, 3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]
        elif in_bedroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_livingroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [1, 2, 3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the pancake",  # 4
            "walk to the microwave",  # 5

            "grab the pancake",  # 6

            "put the pancake in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]

        self.template2action = {
            k: i for i, k in enumerate(self.action_template)
        }
        actions = [self.action_template[i] for i in action_list]
        return {"prompt": text, "action": actions}
    

    def get_state(self, obs):
        return self.obs2text(obs)

    def add_child(self, state, action):
        child = LanguageNode(state = state, last_action=action)
        self._children[action] = child
        child.parent = self
    def get_history_prompt(self):
        return self.history_prompt

class LLMAgent(nn.Module):
    def __init__(self, task = 3, normalization_mode = "token", device = None, epsilon=0.1, alpha = 0.99, gamma =0.99,
                 llm_base_model = "meta-llama/Llama-3.1-8B-Instruct", llm_base_model_path = ""):
        super().__init__()
        self.task = task
        self.base_model = llm_base_model # default Neko-Institute-of-Science/LLaMA-7B-HF / mistralai/Mistral-7B-Instruct-v0.3
        self.model_path = llm_base_model_path
        self.epsilon = epsilon
        self.alpha = alpha # value reward ratio
        self.gamma = gamma # decay rate

        assert (
            self.base_model
        ), "Please specify a --llm-base-model, e.g. --llm-base-model='decapoda-research/llama-7b-hf'"
        assert (
            self.model_path
        ), "Please specify a --llm-base-model-path, e.g. --llm-base-model-path='/data/dengziwei/lcj_test_project/twosome/TWOSOME-main/hf_models/meta-llama/Llama-3.1-8B-Instruct'"
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.normalization_mode = normalization_mode

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, cache_dir=self.model_path)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.model = self._init_llama()

    def _init_llama(self):
        model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        return model

    def select(self, node):
        """
            在当前节点下，选择一个动作指向其一个子节点
            input: node
            output: 选择的动作，对应的值，子节点，动作名称
        """
        action_name_list = []
        action_list = []
        # # get all child-node's action and value
        for action_tmp, child_tmp in node.children.items():
            action_list.append(action_tmp)
            action_name_list.append(child_tmp.last_action)
            
        return torch.tensor([action_list[0]]), node.children[action_list[0]], action_name_list[0]

    def expand(self, obs, node, simulate_envs, history_prompt):
        # 将当前环境观测 obs 转换为文本数据
        text_obs = [self.obs2text(o, history_prompt) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs] # available actions
        action_num = len(action_list[0])
        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]  
        
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # LLM carry
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        action_list = [item for sublist in action_list for item in sublist]
        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)
        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim=-1) - 1  # delete first token
        sequence_length = torch.sum(attention_mask, dim=-1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)
        slices = [gen_logits[i, start - 1:end - 1] for i, (start, end) in enumerate(action_index)]
        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1 == 2
        action_logits = action_logits.reshape(-1, action_num).float()

        actions = torch.exp(action_logits)
        actions = actions / actions.sum()
        # 从值的概率分布采样概率最大的
        argmax = torch.argmax(actions).item()
        # build children node
        envs = copy.deepcopy(simulate_envs)
        argmax_tmp = torch.tensor([argmax], device=self.device)
        next_obs, reward, done, info = envs.step(argmax_tmp.cpu().numpy())
        history_prompt = prompt[0] + '\n' + action_list[argmax] + '\n' 
        print("action_list: ", action_list)
        print("history_prompt: ", history_prompt)
        input()
        child = LanguageNode(parent=node, state=next_obs, last_action=action_list[argmax], history_prompt = history_prompt)
        # node --(action edge)--> child_node
        node.children[argmax] = child

    def simulate(self, node, simulate_envs):
        # 构建环境副本执行模拟操作
        envs = copy.deepcopy(simulate_envs)
        pass

    def obs2text(self, obs, history_prompt):

        text = ""

        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]

        see_pancake = obs[4]
        close_to_pancake = obs[5]
        hold_pancake = obs[6]

        see_microwave = obs[7]
        close_to_microwave = obs[8]
        is_microwave_open = obs[9]

        pancake_in_microwave = obs[10]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        object_text = ""
        action_list = []

        if in_kitchen:

            if not see_pancake:
                object_text += "The pancake is in the microwave. "
            else:
                object_text += "You notice pancake and microwave. "

            if hold_pancake:
                object_text += "Currently, you have grabbed the pancake in hand. "
                if close_to_microwave:
                    object_text += "The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 7, 8, 9]
                else:
                    object_text += "The microwave is not close to you. "
                    action_list = [0, 2, 3, 4, 5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is close to you. "
                    action_list = [0, 2, 3, 5, 6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 8, 9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not close to you. "
                    action_list = [0, 2, 3, 4, 5]
                else:
                    if is_microwave_open:
                        action_list = [0, 2, 3, 8, 9]
                    else:
                        action_list = [0, 2, 3, 9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0, 2, 3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]
        elif in_bedroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_livingroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [1, 2, 3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the pancake",  # 4
            "walk to the microwave",  # 5

            "grab the pancake",  # 6

            "put the pancake in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]

        self.template2action = {
            k: i for i, k in enumerate(self.action_template)
        }
        actions = [self.action_template[i] for i in action_list]
        prompts = history_prompt + text
        return {"prompt": prompts, "action": actions}