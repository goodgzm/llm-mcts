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
        value=None,
        done = False,
        env = False
    ) -> None:
        super().__init__(parent)
        # 当前节点环境观测状态，可以写 state 或 state, 因为 basic MCTS 的 state 并不会有多个
        self.state = state
        self.task = task
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.last_action = last_action   
        self.value = value   
        self._done = done                             
        self._env = env    

    @property
    def obs(self) -> None:
        return self.state

    @property
    def action(self) -> None:
        return self.last_action

    @property
    def done(self) -> None:
        return self._done

    @property
    def env(self) -> None:
        return self._env
    
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

        see_chips = obs[4]
        close_to_chips = obs[5]
        hold_chips = obs[6]
        chips_on_coffeetable = obs[7]

        see_milk = obs[8]
        close_to_milk = obs[9]
        hold_milk = obs[10]
        milk_on_coffeetable = obs[11]

        see_tv = obs[12]
        close_to_tv = obs[13]
        is_face_tv = obs[14]
        is_tv_on = obs[15]

        see_sofa = obs[16]
        close_to_sofa = obs[17]
        is_sit_sofa = obs[18]

        see_coffeetable = obs[19]
        close_to_coffeetable = obs[20]
        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {} "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        ########################################template2####################################
        # template for kitchen
        object_text = ""

        action_list = []

        if in_kitchen:

            if see_chips and see_milk:
                object_text += "and notice chips and milk. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                elif hold_chips and not hold_milk:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not close to you. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips are not close to you. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are close to you. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            5,
                        ]

                    object_text += "Currently, you are not grabbing anything in hand. "

            elif see_chips and not see_milk:
                object_text += "and only notice chips. "

                if hold_chips:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                        ]

            elif not see_chips and see_milk:
                object_text += "and notice milk. "

                if hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                        ]

            else:
                object_text += "and notice nothing. "

                action_list = [
                    0,
                    2,
                    3,
                ]

        elif in_livingroom:

            object_text += "and you notice a coffee table, a TV and a sofa. "

            assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "You are next to more than one object from coffee table, TV and sofa."
            assert see_coffeetable + see_tv + see_sofa >= 3, "You don't see coffee table, TV and sofa."

            if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                object_text += "They are not close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "
                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "
                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                action_list = [
                    1,
                    2,
                    3,
                    6,
                    7,
                    8
                ]

            if close_to_coffeetable:

                if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                    object_text += "The TV is not close to you. "
                else:
                    object_text += "The coffee table is close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        7,
                        8,
                        11,
                        12
                    ]
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += "Currently, you have grabbed the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            12
                        ]

                    else:
                        object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    if not milk_on_coffeetable:
                        object_text += "Currently, you have grabbed the chips in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            11
                        ]

                    else:
                        object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                    ]

            if close_to_tv:
                if is_tv_on:
                    object_text += "The sofa is not close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                    ]

                else:
                    object_text += "The TV is close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                        13,
                        14
                    ]

            if close_to_sofa:

                if not is_sit_sofa:
                    object_text += "The sofa is close to you. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                            15,
                            16
                        ]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                        ]

                else:
                    object_text += "You are sitting on the sofa. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]

        elif in_bedroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]

        elif in_bathroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]

        text += object_text

        # template for target
        target_template = "In order to enjoy the chips and the milk while watching TV, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the chips",  # 4
            "walk to the milk",  # 5
            'walk to the coffee table',  # 6
            'walk to the TV',  # 7
            'walk to the sofa',  # 8

            "grab the chips",  # 9
            "grab the milk",  # 10

            'put the chips on the coffee table',  # 11
            'put the milk on the coffee table',  # 12

            "turn on the TV",  # 13
            "turn off the TV",  # 14

            "sit on the sofa",  # 15
            "stand up from the sofa"  # 16
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

    def select(self, nodes, b):
        """
            在当前节点下，选择一个动作指向其一个子节点
            input: node
            output: 选择的动作，对应的值，子节点，动作名称
        """
        action_name_list = []
        value_list = []
        len_nodes = len(nodes)
        for child_tmp in nodes:
            value_list.append(child_tmp.value)
            action_name_list.append(child_tmp.last_action)
        value_list = torch.stack(value_list)
        value_list = value_list.view(-1)
        values = value_list / value_list.sum()
        max_id_list = torch.argsort(values, descending=True)
        # print("#############max_id_list", max_id_list)
        # print("##############nodes_list", [nodes[max_id_list[_]] for _ in range(b)])
        return [nodes[max_id_list[_]] for _ in range(b)]

    def expand(self, obs, node, is_stochastic):
        simulate_envs = node.env
        # 将当前环境观测 obs 转换为文本数据
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs] # available actions
        if node.parent is not None:
            tb_p = [self.obs2text(o) for o in node.parent.get_state()]
            p_p = [o["prompt"] for o in tb_p]
            a_p = [o["action"] for o in tb_p]
        #     print("###########parent_id", node.parent)
        #     print("#############parent_prompt", p_p)
        #     print("#############parent_action_list", a_p)
        # print("###########id", node)
        # print("#############prompt", prompt)
        # print("#############action_list", action_list)
        # print("____________________________________________________")
        # input()
        # build children node
        for id, action in enumerate(action_list[0]):
            envs = copy.deepcopy(simulate_envs)
            id_tmp = torch.tensor([id], device=self.device)
            next_obs, reward, done, info = envs.step(id_tmp.cpu().numpy())
            if 'chop' in action and is_stochastic:  ###???????
                envs = simulate_envs
                next_obs = node.obs
                done = node.done

            child = LanguageNode(parent=node, state=next_obs, env = envs, last_action = action, done = done)
            child.to_value(self.get_value(child))
            node.children.append(child)
            # print("#############node.children", len(node.children))
    
    def get_value(self, node):
        parent = node.parent
        if not parent:
            raise ValueError("Root has no value!!")
        text_obs_parent = [self.obs2text(o, have_target = False) for o in parent.obs]
        text_obs_node = [self.obs2text(o, have_target = False) for o in node.obs]
        prompt_node = ["" for _ in range(len(text_obs_node))]
        prompt_parent = ["" for _ in range(len(text_obs_parent))]
        for ip, (o1, o2) in enumerate(zip(text_obs_parent, text_obs_node)):
            for p1, p2 in zip(o1["prompt"], o2["prompt"]):
                prompt_parent[ip] += p1
                prompt_node[ip] += p2
        
        prompt = []
        overall_goal = "overall_goal : To enjoy the chips and the milk while watching TV"
        quetions = "To achieve the overall goal, is it reasonable to move from state 1 to state 2 through actions?"
        
        for p1, a, p2 in zip(prompt_parent, node.action, prompt_node):
            prompt += [overall_goal + '\n' + "state 1: " + p1 + '\n' + 'action: ' + a + '\n' + "state 2: " + p2 + '\n' + quetions]
        
        action_list_yn = [["no", "yes"] for _ in range(len(prompt))]
        action_num = len(action_list_yn[0])
        sequence = []
        for p, ac in zip(prompt, action_list_yn):
            sequence += [p + " " + a for a in ac]  
        
        inputs = self.tokenizer(sequence, return_tensors = "pt", padding = True)
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask = attention_mask, temperature = 0.7)

        action_list_yn = [item for sublist in action_list_yn for item in sublist]
        self.action_list_ids = self.tokenizer(action_list_yn, return_tensors = "pt", padding = True)
        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim = -1) - 1
        sequence_length = torch.sum(attention_mask, dim = -1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)] 

        logits = torch.log_softmax(outputs.logits, dim = -1)
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)
        slices = [gen_logits[i, start - 1:end - 1] for i, (start, end) in enumerate(action_index)]
        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list_yn]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1 == 2
        
        action_logits = action_logits.reshape(-1, action_num).float()
        
        actions = torch.exp(action_logits)
        
        actions = actions / actions.sum()
        
        return actions[:, 1]

    def simulate(self, node, simulate_envs):
        # 构建环境副本执行模拟操作
        envs = copy.deepcopy(simulate_envs)
        pass

    def obs2text(self, obs, have_target = True):
        text = ""
        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]

        see_chips = obs[4]
        close_to_chips = obs[5]
        hold_chips = obs[6]
        chips_on_coffeetable = obs[7]

        see_milk = obs[8]
        close_to_milk = obs[9]
        hold_milk = obs[10]
        milk_on_coffeetable = obs[11]

        see_tv = obs[12]
        close_to_tv = obs[13]
        is_face_tv = obs[14]
        is_tv_on = obs[15]

        see_sofa = obs[16]
        close_to_sofa = obs[17]
        is_sit_sofa = obs[18]

        see_coffeetable = obs[19]
        close_to_coffeetable = obs[20]
        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {} "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        ########################################template2####################################
        # template for kitchen
        object_text = ""

        action_list = []

        if in_kitchen:

            if see_chips and see_milk:
                object_text += "and notice chips and milk. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                elif hold_chips and not hold_milk:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not close to you. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips are not close to you. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are close to you. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            5,
                        ]

                    object_text += "Currently, you are not grabbing anything in hand. "

            elif see_chips and not see_milk:
                object_text += "and only notice chips. "

                if hold_chips:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                        ]

            elif not see_chips and see_milk:
                object_text += "and notice milk. "

                if hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                        ]

            else:
                object_text += "and notice nothing. "

                action_list = [
                    0,
                    2,
                    3,
                ]

        elif in_livingroom:

            object_text += "and you notice a coffee table, a TV and a sofa. "

            assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "You are next to more than one object from coffee table, TV and sofa."
            assert see_coffeetable + see_tv + see_sofa >= 3, "You don't see coffee table, TV and sofa."

            if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                object_text += "They are not close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "
                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "
                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                action_list = [
                    1,
                    2,
                    3,
                    6,
                    7,
                    8
                ]

            if close_to_coffeetable:

                if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                    object_text += "The TV is not close to you. "
                else:
                    object_text += "The coffee table is close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        7,
                        8,
                        11,
                        12
                    ]
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += "Currently, you have grabbed the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            12
                        ]

                    else:
                        object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    if not milk_on_coffeetable:
                        object_text += "Currently, you have grabbed the chips in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            11
                        ]

                    else:
                        object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                    ]

            if close_to_tv:
                if is_tv_on:
                    object_text += "The sofa is not close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                    ]

                else:
                    object_text += "The TV is close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                        13,
                        14
                    ]

            if close_to_sofa:

                if not is_sit_sofa:
                    object_text += "The sofa is close to you. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                            15,
                            16
                        ]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                        ]

                else:
                    object_text += "You are sitting on the sofa. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]

        elif in_bedroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]

        elif in_bathroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]

        text += object_text

        if have_target == True:
            # template for target
            target_template = "In order to enjoy the chips and the milk while watching TV, "
            text += target_template
            # template for next step    
            next_step_text = "your next step is to"
            text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the chips",  # 4
            "walk to the milk",  # 5
            'walk to the coffee table',  # 6
            'walk to the TV',  # 7
            'walk to the sofa',  # 8

            "grab the chips",  # 9
            "grab the milk",  # 10

            'put the chips on the coffee table',  # 11
            'put the milk on the coffee table',  # 12

            "turn on the TV",  # 13
            "turn off the TV",  # 14

            "sit on the sofa",  # 15
            "stand up from the sofa"  # 16
        ]

        self.template2action = {
            k: i for i, k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]
    
        # return {"prompt": text, "action": actions}

        return {"prompt": text, "action": actions}