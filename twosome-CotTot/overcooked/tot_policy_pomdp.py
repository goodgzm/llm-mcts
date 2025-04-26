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
        self._children = []
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

    # @property
    # def _state(self) -> None:
    #     return self.state
    
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
        if self.task == 3:
            obs = obs.tolist()
            action_list = [
                "pick up the tomato",
                "pick up the lettuce",
                "pick up the onion",
                "take the empty bowl",
                "walk to the first cutting board",
                "walk to the second cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0, 0, 0]
            ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
            raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
            chopped = [False, False, False]
            ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
            sentences = ["There are two fixed cutting boards in the room."]

            item = []
            item_index = []
            agent_pos = obs[17:19]
            first_cutting_board_pos = [1, 0]
            second_cutting_board_pos = [2, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos,
                        "in_second_cutting_board": second_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": [], "in_second_cutting_board": []}

            for i in range(4):
                pos = obs[3 * i: 3 * i + 2]
                if pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)

                if i < 3 and obs[3 * i + 2] == 3:
                    chopped[i] = True

                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)

                        if len(overlay[k]) > 1:
                            action_list[3] = "take the bowl"

            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."
            elif len(item) == 3:
                template = "You notice {}, {} and {} on the different tables."
            elif len(item) == 4:
                template = "You notice {}, {}, {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize())

            cutting_board_index = ["first", "second"]
            cutting_board_name = ["in_first_cutting_board", "in_second_cutting_board"]
            for cindex in range(2):
                if len(overlay[cutting_board_name[cindex]]) == 1:
                    id = overlay[cutting_board_name[cindex]][0]
                    template = "{} is on the {} cutting board."
                    if id == 3:
                        sentences.append(template.format("a bowl", cutting_board_index[cindex]).capitalize())
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id],
                                                             cutting_board_index[cindex]).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id],
                                                             cutting_board_index[cindex]).capitalize())
                        if agent_pos == [cindex + 1, 1]:
                            action_list[-1] = "chop the " + raw_ingredient[id]

                elif len(overlay[cutting_board_name[cindex]]) > 1:
                    in_plate_item = overlay[cutting_board_name[cindex]][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "A bowl containing chopped {} is on the {} cutting board."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "A bowl containing chopped {} and {} is on the {} cutting board."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "A bowl containing chopped {}, {} and {} is on the {} cutting board."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item],
                                                                cutting_board_index[cindex]).capitalize())

                    # in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            # in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the {} cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
                "put the lettuce in the bowl",
                "put the onion in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the {} cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are standing in front of the {} cutting board, carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format(cutting_board_index[cindex], "a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
                    else:
                        if chopped[id]:
                            sentences.append(template.format(cutting_board_index[cindex],
                                                             "a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format(cutting_board_index[cindex],
                                                             "an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[4] = action_template.format(raw_ingredient[id], "first")
                            action_list[5] = action_template.format(raw_ingredient[id], "second")
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} in hand."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} and {} in hand."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {}, {} and {} in hand."

                    sentences.append(full_plate_template.format(cutting_board_index[cindex],
                                                                *[raw_ingredient[id] for id in
                                                                  in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[4] = action_template.format(raw_ingredient[id], "first")
                            action_list[5] = action_template.format(raw_ingredient[id], "second")
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."

                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
            sentences.append("To serve the dish of a bowl only containing chopped tomato and lettuce, you should first")
        elif self.task == 0:
            obs = obs.tolist()

            action_list = [
                "pick up the tomato",
                "take the bowl",
                "walk to the cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0]
            ingredient = ["a tomato", "a bowl"]
            raw_ingredient = ["tomato", "bowl"]
            chopped = [False]
            ori_pos = [[0, 5], [6, 5]]
            sentences = ["There is a fixed cutting board in the room."]
            in_plate = [False, False, False]

            item = []
            item_index = []
            plate_pos = obs[3:5]
            agent_pos = obs[9:11]
            first_cutting_board_pos = [1, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": []}

            for i in range(2):
                pos = obs[3 * i: 3 * i + 2]
                if pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)

                if i < 1 and obs[3 * i + 2] == 3:
                    chopped[i] = True

                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize())

            cutting_board_index = ["first"]
            cutting_board_name = ["in_first_cutting_board"]

            cindex = 0
            if len(overlay[cutting_board_name[cindex]]) == 1:
                id = overlay[cutting_board_name[cindex]][0]
                template = "{} is on the cutting board."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                    if agent_pos == [cindex + 1, 1]:
                        action_list[-1] = "chop the " + raw_ingredient[id]


            elif len(overlay[cutting_board_name[cindex]]) > 1:

                full_plate_template = "a bowl containing a chopped tomato is on the cutting board."
                sentences.append(full_plate_template.capitalize())

                # in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            # in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    id = overlay["in_agent"][0]
                    action_list[3] = "serve the dish"
                    template = "Currently you are standing in front of the cutting board, carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id])
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[2] = action_template.format(raw_ingredient[id])
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the cutting board, carrying a bowl containing chopped {} in hand."
                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[3] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id])
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[2] = action_template.format(raw_ingredient[id])
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."
                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")

            sentences.append("To serve the dish of a bowl only containing chopped tomato, you should first")

        return {"prompt": " ".join(sentences), "action": action_list}
    
    def get_state(self):
        return self.state

    def add_child(self, state, action):
        child = LanguageNode(state = state, last_action=action)
        self._children[action] = child
        child.parent = self

    def to_value(self,  value):
        self.value = value
    
    

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

    def expand(self, obs, node, is_stochastic = 0):
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
            if 'chop' in action and is_stochastic:
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
        overall_goal = ""
        if self.task == 3:
            overall_goal = "overall_goal : To serve the dish of a bowl only containing chopped tomato and lettuce"
        elif self.task == 0:
            overall_goal = "overall_goal : To serve the dish of a bowl only containing chopped tomato"
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
        if self.task == 3:
            obs = obs.tolist()
            action_list = [
                "pick up the tomato",
                "pick up the lettuce",
                "pick up the onion",
                "take the empty bowl",
                "walk to the first cutting board",
                "walk to the second cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0, 0, 0]
            ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
            raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
            chopped = [False, False, False]
            ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
            sentences = ["There are two fixed cutting boards in the room."]

            item = []
            item_index = []
            agent_pos = obs[17:19]
            first_cutting_board_pos = [1, 0]
            second_cutting_board_pos = [2, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos,
                        "in_second_cutting_board": second_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": [], "in_second_cutting_board": []}

            for i in range(4):
                pos = obs[3 * i: 3 * i + 2]
                if pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)

                if i < 3 and obs[3 * i + 2] == 3:
                    chopped[i] = True

                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)

                        if len(overlay[k]) > 1:
                            action_list[3] = "take the bowl"

            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."
            elif len(item) == 3:
                template = "You notice {}, {} and {} on the different tables."
            elif len(item) == 4:
                template = "You notice {}, {}, {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize())

            cutting_board_index = ["first", "second"]
            cutting_board_name = ["in_first_cutting_board", "in_second_cutting_board"]
            for cindex in range(2):
                if len(overlay[cutting_board_name[cindex]]) == 1:
                    id = overlay[cutting_board_name[cindex]][0]
                    template = "{} is on the {} cutting board."
                    if id == 3:
                        sentences.append(template.format("a bowl", cutting_board_index[cindex]).capitalize())
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id],
                                                             cutting_board_index[cindex]).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id],
                                                             cutting_board_index[cindex]).capitalize())
                        if agent_pos == [cindex + 1, 1]:
                            action_list[-1] = "chop the " + raw_ingredient[id]

                elif len(overlay[cutting_board_name[cindex]]) > 1:
                    in_plate_item = overlay[cutting_board_name[cindex]][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "A bowl containing chopped {} is on the {} cutting board."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "A bowl containing chopped {} and {} is on the {} cutting board."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "A bowl containing chopped {}, {} and {} is on the {} cutting board."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item],
                                                                cutting_board_index[cindex]).capitalize())

                    # in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            # in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the {} cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
                "put the lettuce in the bowl",
                "put the onion in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the {} cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are standing in front of the {} cutting board, carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format(cutting_board_index[cindex], "a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
                    else:
                        if chopped[id]:
                            sentences.append(template.format(cutting_board_index[cindex],
                                                             "a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format(cutting_board_index[cindex],
                                                             "an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[4] = action_template.format(raw_ingredient[id], "first")
                            action_list[5] = action_template.format(raw_ingredient[id], "second")
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} in hand."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} and {} in hand."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {}, {} and {} in hand."

                    sentences.append(full_plate_template.format(cutting_board_index[cindex],
                                                                *[raw_ingredient[id] for id in
                                                                  in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first")
                        action_list[5] = action_template.format(raw_ingredient[id], "second")
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[4] = action_template.format(raw_ingredient[id], "first")
                            action_list[5] = action_template.format(raw_ingredient[id], "second")
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."

                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first")
                    action_list[5] = action_template.format("bowl", "second")
            if have_target == True:
                sentences.append("To serve the dish of a bowl only containing chopped tomato and lettuce, you should first")
        elif self.task == 0:
            obs = obs.tolist()

            action_list = [
                "pick up the tomato",
                "take the bowl",
                "walk to the cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0]
            ingredient = ["a tomato", "a bowl"]
            raw_ingredient = ["tomato", "bowl"]
            chopped = [False]
            ori_pos = [[0, 5], [6, 5]]
            sentences = ["There is a fixed cutting board in the room."]
            in_plate = [False, False, False]

            item = []
            item_index = []
            plate_pos = obs[3:5]
            agent_pos = obs[9:11]
            first_cutting_board_pos = [1, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": []}

            for i in range(2):
                pos = obs[3 * i: 3 * i + 2]
                if pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)

                if i < 1 and obs[3 * i + 2] == 3:
                    chopped[i] = True

                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize())

            cutting_board_index = ["first"]
            cutting_board_name = ["in_first_cutting_board"]

            cindex = 0
            if len(overlay[cutting_board_name[cindex]]) == 1:
                id = overlay[cutting_board_name[cindex]][0]
                template = "{} is on the cutting board."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize())
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                    if agent_pos == [cindex + 1, 1]:
                        action_list[-1] = "chop the " + raw_ingredient[id]


            elif len(overlay[cutting_board_name[cindex]]) > 1:

                full_plate_template = "a bowl containing a chopped tomato is on the cutting board."
                sentences.append(full_plate_template.capitalize())

                # in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            # in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    id = overlay["in_agent"][0]
                    action_list[3] = "serve the dish"
                    template = "Currently you are standing in front of the cutting board, carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id])
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[2] = action_template.format(raw_ingredient[id])
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the cutting board, carrying a bowl containing chopped {} in hand."
                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize())

                elif len(overlay["in_agent"]) == 1:
                    action_list[3] = "serve the dish"
                    id = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id])
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize())
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize())
                            action_list[2] = action_template.format(raw_ingredient[id])
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."
                    sentences.append(
                        full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl")
            if have_target == True:
                sentences.append("To serve the dish of a bowl only containing chopped tomato, you should first")

        return {"prompt": sentences, "action": action_list}


    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj
