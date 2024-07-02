
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field, HttpUrl, EmailStr
from typing import List, Optional
import json
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import dotenv
from typing import Optional
import csv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


class Resume(BaseModel):
    # base information
    name: Optional[str] = Field(None, description="姓名", example="张三")
    sex: Optional[str] = Field(None, description="性别", example="男")
    age: Optional[int] = Field(None, description="到2024年的年龄", example="23")
    email: Optional[str] = Field(None, description="电子邮箱", example="zhangsan@example.com")
    phone: Optional[str] = Field(None, description="联系电话", example="13800000000")
    location: Optional[str] = Field(None, description="居住地", example="北京")
    target_workplace: Optional[str] = Field(None, description="目标工作地", example="上海")
    personal_url: Optional[str] = Field(None, description="LinkedIn 个人页面", example="https://www.linkedin.com/in/zhangsan")

    # educational background
    bachelor: Optional[str] = Field(None, description="本科学校", example="xx大学")
    master: Optional[str]= Field(None, description="硕士学校", example="xx大学")
    doctorate: Optional[str] = Field(None, description="博士学校", example="xx大学")
    bachelor_major: Optional[str]= Field(None, description="本科专业", example="电子信息工程")
    master_major: Optional[str] = Field(None, description="硕士专业", example="电子信息工程")
    doctorate_major: Optional[str] = Field(None, description="博士专业", example="电子信息工程")
    bachelor_grad_date: Optional[str] = Field(None, description="本科毕业时间", example="2013")
    master_grad_date: Optional[str] = Field(None, description="硕士毕业时间", example="2015")
    doctorate_grad_date: Optional[str] = Field(None, description="博士毕业时间", example="2018")
    honors: Optional[str] = Field(None, description="个人荣誉", example="北京大学优秀毕业生, 国家奖学金, 北京市优秀毕业生")
    certifications: Optional[str] = Field(None, description="技能证书", example="大学英语六级, 计算机一级考试,普通话一级乙等")
    language_levels: Optional[str] = Field(None, description="语言等级", example="六级: 666, 四级: 688, 托福: 110")
    
    # working experience
    experience: Optional[str] = Field(None, description="个人经历", example="上海中建东孚投资发展有限公司 人力资源岗。")
    skills: Optional[str] = Field(None, description="技能", example="Python, 机器学习, 数据分析")
    projects: Optional[str] = Field(None, description="项目经历", example="上海中建东孚投资发展有限公司 人力资源岗。")
    expectations: Optional[str] = Field(None, description="期望", example="期望薪资40万, 职位为全栈开发工程师")

 
    # personality 
    self_evaluation: Optional[str] = Field(None, description="自我评价", example="乐观开朗，积极大方，善于组织和规划，乐于学习新知识...")
    hobbies: Optional[str] = Field(None, description="爱好", example="阅读, 健身, 游泳")
    motivation: Optional[str] = Field(None, description="工作动机", example="提升专业技能，掌握前沿技术，成为行业专家")
    summary: Optional[str] = Field(None, description="个人简介", example="有5年软件开发经验, 擅长机器学习和数据分析。")

def generate_cv_template(cv_path, save_template_path):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=Resume) 

    # embeddings_model  = OpenAIEmbeddings(model="text-embedding-3-large")

    embeddings_model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda"
    )

    df = pd.read_csv(cv_path)
    # print(len(df)) #18839
    column_names = ['uuid', 'CV_content', 'template', 'embedding']
    with open(save_template_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for idx, row in df.iterrows():
            # if idx != 737:
            #     continue
            cv_content = row['CV_content']
            try:
                response = runnable.invoke({"text": cv_content})
            except Exception as e:
                print(e)
                print(f"write {idx}-th cv to {save_template_path} fail!")
                continue
            data_json = json.dumps(dict(response), ensure_ascii=False, indent=2)
            text = ""
            for key, value in dict(response).items():
                if value != None and value != 'null':
                    text += str(value) + " "

            # embedding_model = embeddings.embed_documents([text])[0]

            embedding = embeddings_model.encode(
                [text],
                show_progress_bar=True,
            )[0]

            new_row = []
            new_row.append(row['uuid'])
            new_row.append(row['CV_content'])
            new_row.append(data_json)
            new_row.append(embedding)
            writer.writerow(new_row)
            print(f"write {idx}-th cv to {save_template_path} success!")

            # if idx == 3:
            #     break

    print("write cv template success!")


if __name__ == '__main__':

    cv_path = '/autodl-fs/data/wang/falcon/csvfile/CV.csv'
    save_template_path = '/autodl-fs/data/song/jd_cv_gpt4_template/cv_template_embedding.csv'
    # save_template_path = 'cv_template_embedding.csv'

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                "If you do not know the value of an attribute asked to extract, "
                "return null for the attribute's value.",
            ),
            ("human", "{text}"),
        ]
    )
    # llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
    runnable = prompt | llm.with_structured_output(schema=Resume) 

    # embeddings_model  = OpenAIEmbeddings(model="text-embedding-3-large")

    embeddings_model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cuda"
    )

    df = pd.read_csv(cv_path)
    # print(len(df)) #18839
    column_names = ['uuid', 'CV_content', 'template', 'embedding']
    with open(save_template_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        for idx, row in df.iterrows():
            # if idx != 737:
            #     continue
            cv_content = row['CV_content']
            try:
                response = runnable.invoke({"text": cv_content})
            except Exception as e:
                print(e)
                print(f"write {idx}-th cv to {save_template_path} fail!")
                continue
            data_json = json.dumps(dict(response), ensure_ascii=False, indent=2)
            text = ""
            for key, value in dict(response).items():
                if value != None and value != 'null':
                    text += str(value) + " "

            # embedding_model = embeddings.embed_documents([text])[0]

            embedding = embeddings_model.encode(
                [text],
                show_progress_bar=True,
            )[0]

            new_row = []
            new_row.append(row['uuid'])
            new_row.append(row['CV_content'])
            new_row.append(data_json)
            new_row.append(embedding)
            writer.writerow(new_row)
            print(f"write {idx}-th cv to {save_template_path} success!")

            # if idx == 3:
            #     break

    print("write cv template success!")
            





