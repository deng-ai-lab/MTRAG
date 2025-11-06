
CAPTION_QUESTIONS = [
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please provide a thorough description of the this image',
    'Please provide a thorough description of the this image.',
    'Please describe in detail the contents of the image.',
    'Please describe in detail the contents of the image',
    'Could you give a comprehensive explanation of what can be found within this picture?',
    'Could you give me an elaborate explanation of this picture?',
    'Could you provide me with a detailed analysis of this photo?',
    'Could you please give me a detailed description of the image?',
    'Can you provide a thorough description of the this image?',
    'Please describe in detail the contents of the image',
    'Please describe in detail the contents of the image.',
    'Can you give a comprehensive explanation of this photo',
    'Please provide an elaborate explanation of this picture.',
    'Please provide an elaborate explanation of this picture',
    'Could you provide me with a detailed analysis of this photo',
]

REGION_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give ablout the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represtented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

REGION_GROUP_QUESTIONS = [
    'Could you please give me a detailed description of these areas <region>?',
    'Can you provide a thorough description of the regions <region> in this image?',
    'Please describe in detail the contents of the boxed areas <region>.',
    'Could you give a comprehensive explanation of what can be found within <region> in the picture?',
    'Could you give me an elaborate explanation of the <region> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <region> in this photo?',
    'Help me understand the specific locations labeled <region> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <region> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <region> in this photo?',
    'What are the specific features of the areas marked <region> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <region> in this image?',
    'What can you tell me about the areas labeled <region> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <region> in this photo?',
    'I am interested in learning more about the regions marked <region> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <region> in this photo?',
    'What is the significance of the regions labeled <region> in this picture?',
    'I would like to know more about the specific locations designated <region> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <region> in this photo?',
    'What specific features can you tell me about the areas identified by <region> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <region> in this image?',
    'Can you provide a detailed account of the regions designated <region> in this photo?',
    'I am curious about the areas marked <region> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <region> in this image?',
    'Could you please provide a detailed description of the regions labeled <region> in this photo?',
    'What can you tell me about the features of the areas designated <region> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <region> in this image?',
    'I would like to know more about the specific locations identified by <region> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <region> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <region> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <region> in this photo?'
]

REGION_SEPARATE_QUESTIONS = [
    'Could you please give me a detailed description of each of the following areas: <region>?',
    'Can you provide a thorough, separate description for each region in <region> shown in this image?',
    'Please describe in detail the contents of the boxed areas: <region>, specifying each individually.',
    'Could you give a comprehensive explanation of what can be found within each part of <region> in the picture?',
    'Could you give an elaborate explanation for each region in <region> separately within this picture?',
    'Can you provide a comprehensive, distinct description of each area identified in <region> in this photo?',
    'Help me understand the specific locations labeled in <region> in this picture, detailing each one separately.',
    'What is the detailed information about each of these areas marked as <region> in this image?',
    'Could you provide me with a detailed analysis of each region in <region> in this photo, separately?',
    'What specific features can you describe in detail for each area marked as <region> in this picture?',
    'Could you elaborate on each of the regions listed as <region> in this image, addressing them separately?',
    'What can you tell me about each of the areas labeled as <region> in this picture?',
    'Can you provide a thorough analysis of each specific location in <region> in this photo?',
    'I am interested in learning more about the regions marked as <region> in this image. Could you provide separate information for each?',
    'Could you please provide a detailed description for each area identified in <region> in this photo, one by one?',
    'What is the significance of each of the regions labeled as <region> in this picture, separately?',
    'I would like to know more about each specific location in <region> in this image. Could you provide separate details for each area?',
    'Can you provide a detailed breakdown of each separate region listed as <region> in this photo?',
    'What specific features can you tell me about each individual area marked in <region> in this picture?',
    'Could you please provide a comprehensive explanation of each location labeled in <region> in this image?',
    'Can you provide a detailed account of each distinct region in <region> in this photo?',
    'I am curious about the areas marked as <region> in this picture. Could you provide a detailed, separate analysis for each?',
    'What important details can you tell me about each specific location identified as <region> in this image?',
    'Could you please provide a detailed description of each region labeled in <region> in this photo?',
    'What can you tell me about the features of each area designated as <region> in this picture?',
    'Could you provide a comprehensive overview of each region marked in <region> in this image, one by one?',
    'I would like to know more about each specific location listed in <region> in this photo. Could you provide more information for each?',
    'What detailed information do you have for each area labeled as <region> in this picture?',
    'Could you provide me with a thorough analysis of each region listed as <region> in this image?',
    'Can you provide a detailed explanation of each specific location marked as <region> in this photo, individually?'
]

HUMANIZED_INTRODUCTIONS = [
    "I'm happy to help! Here’s a detailed description of each specified area:",
    "Thank you for reaching out! Below is a comprehensive look at each region you mentioned:",
    "I'm glad to answer your question! Here’s a breakdown of each region as requested:",
    "Absolutely, I’m here to provide the details. Here’s a look at each area individually:",
    "It’s my pleasure to assist! Below, you’ll find a detailed description for each specified area:",
    "Certainly! Here’s an in-depth look at the areas you've highlighted:",
    "Thank you for your question! Let’s go over each region in detail as follows:",
    "I’m delighted to assist! Here’s a thorough explanation of each region you’ve specified:",
    "Glad to help! Here’s what we see in each of the regions listed:",
    "I'm here to provide clarity! Let’s dive into the details for each specified area:"
]

REGION_TEMPLATES = [
    "In region{num}, you'll find {content}.",
    "Looking at region{num}, we can see {content}.",
    "Moving to region{num}, there is {content}.",
    "As for region{num}, it contains {content}.",
    "Focusing on region{num}, we observe {content}.",
    "Region{num} presents {content}.",
    "In region{num}, {content} stands out."
]
GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

MO_SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What are the {class_name} in this image? Please respond with segmentation masks.",
    "What are the {class_name} in this image? Please output segmentation masks.",

    "Could you provide segmentation masks for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where are the {class_name} in this picture? Please respond with segmentation masks.",
    "Can you highlight the {class_name} in this image with segmentation masks?",

    "Could you provide segmentation masks for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where are the {class_name} in this picture? Please respond with segmentation masks.",
    "Can you highlight the {class_name} in this image with segmentation masks?",
]
MO_SEG_ANSWER_TEMPLATE = [
    "Here are the segmentation masks for {class_name} in the image.",
    "The segmentation masks for {class_name} are provided below.",
    "I have identified and segmented {class_name} in the image. Here are the segmentation masks.",
    "The segmentation results for {class_name} are as follows.",
    "Below are the segmentation masks highlighting the {class_name} in this image.",
    "The requested segmentation masks for {class_name} have been generated.",
    "Here are the segmented regions for {class_name} in the image.",
    "I have processed the image and extracted segmentation masks for {class_name}."
]






RIO_QUESTIONS=[
    "{sent} Can you identify the thing? Don't leave out any instances of the thing. Please output segmentation mask.",
    "{sent} Show me the thing. Please respond with segmentation mask.",
    "{sent} What is the thing used in this scenario? Please respond with segmentation mask.",
    "{sent} Can you tell me what the thing refers to? Please output segmentation mask.",
    "{sent} Help me recognize the thing mentioned here. Please respond with segmentation mask.",
    "{sent} What object fits the description of the thing? Please output segmentation mask.",
    "{sent} What do you think the thing is? Please output segmentation mask.",
    "{sent} Can you determine what the thing is based on the description? Please respond with segmentation mask.",
    "{sent} Please find the thing based on the given context. Please respond with segmentation mask.",
    "{sent} What do you believe the thing represents? Please output segmentation mask."
]
RIO_ANSWERS = [
    "I'm thinking {object} can be used to do that.",
    "Taking a closer look, it would seem the thing is {object}.",
    "It looks like the thing you need is {object}.",
    "I believe {object} is the most suitable option here.",
    "From the description, the thing appears to be {object}.",
    "Considering the context, {object} would be the correct identification.",
    "Based on the details, {object} is likely the thing you're looking for.",
    "It makes sense that {object} is the thing described here.",
    "I would say {object} best matches the thing in the sentence.",
    "Analyzing the context, {object} seems to be the best fit for the thing.",
]