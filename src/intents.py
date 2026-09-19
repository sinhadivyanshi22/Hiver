"""Intent definitions for AmazonHelp support queries.

Intents are defined from the data: each intent is illustrated with a handful of
real, representative queries mined from the dataset (the "seeds"). These seeds
power a few-shot embedding centroid classifier and are also used to bootstrap
self-training.

Escalation tendency per intent encodes the brand's real behaviour on Twitter:
Amazon's Twitter handle routinely escalates anything that needs account / order /
billing details to a private channel or human (it cannot act on PII over public
Twitter), while simple product info and acknowledgements can be auto-handled.
"""
from __future__ import annotations

from typing import TypedDict


class IntentSpec(TypedDict):
    id: str
    name: str
    description: str
    escalate_default: bool  # does the brand normally escalate this?
    seeds: list[str]


INTENTS: dict[str, IntentSpec] = {
    "order_status": IntentSpec(
        id="order_status",
        name="Order status / delivery",
        description=(
            "Customer asks where an order is, whether it shipped, delivery "
            "timing, tracking, or shipping delays."
        ),
        escalate_default=True,  # often needs order number -> private channel
        seeds=[
            "Where is my order?",
            "Has my order shipped?",
            "What is the status of my order?",
            "The tracking says delivered but I have not received anything.",
            "How long does delivery take?",
            "Why is my package delayed?",
            "When will it arrive?",
            "What does shipping mean for this item?",
            "1 hour to process amazon pay balance into the account",
            "why offer pre 1pm delivery if you're not actually going to abide by it?",
            "My package is stuck in transit.",
        ],
    ),
    "return_refund": IntentSpec(
        id="return_refund",
        name="Return / refund",
        description="Customer wants to return an item, request a refund, or report a missing refund.",
        escalate_default=True,
        seeds=[
            "I want to return this item.",
            "How do I get a refund?",
            "I have not received my refund yet.",
            "I received the wrong item and want a refund.",
            "Can I return something I bought?",
            "What is the refund policy?",
            "How do I print a return label?",
            "I started a return but it shows no progress.",
        ],
    ),
    "account_access": IntentSpec(
        id="account_access",
        name="Account access / login",
        description="Login, password reset, account creation, account lockouts, Prime sign-up.",
        escalate_default=True,  # needs account details
        seeds=[
            "I cannot log in to my account.",
            "I forgot my password.",
            "How do I reset my password?",
            "My account is locked.",
            "I can't access my account.",
            "I am not able to login",
            "I can't log in my account",
            "Why can't I sign in to my account?",
            "my account has been hacked, I cannot login",
        ],
    ),
    "billing_payment": IntentSpec(
        id="billing_payment",
        name="Billing / payment",
        description="Payment method issues, declined cards, charges, balances, Amazon Pay.",
        escalate_default=True,  # needs payment details
        seeds=[
            "My card was declined.",
            "I was charged twice for the same order.",
            "My payment method is not working.",
            "I have a balance that will not process.",
            "The payment says failed but my bank was charged.",
            "Why was I charged for Prime?",
        ],
    ),
    "technical_issue": IntentSpec(
        id="technical_issue",
        name="Technical issue",
        description="Website/app errors, can't load pages, download problems, error messages.",
        escalate_default=False,
        seeds=[
            "I get an error when I try to check out.",
            "The app keeps crashing.",
            "I can't download the file.",
            "The page is not loading.",
            "There was an error trying to send your e-mail.",
            "It keeps logging me out of the app.",
            "The checkout button does nothing.",
        ],
    ),
    "damage_defective": IntentSpec(
        id="damage_defective",
        name="Damaged / defective / wrong item",
        description="Received a damaged, defective, broken, or incorrect item.",
        escalate_default=True,
        seeds=[
            "I received a damaged item.",
            "The product is defective.",
            "I got the wrong item.",
            "My new fridge came broken.",
            "The item arrived damaged.",
            "I got a used item instead of new.",
            "The screen arrived cracked.",
        ],
    ),
    "product_inquiry": IntentSpec(
        id="product_inquiry",
        name="Product / service inquiry",
        description="Pre-purchase questions: availability, features, shipping to a country, costs.",
        escalate_default=False,  # usually auto-handled with a quick factual reply
        seeds=[
            "Do you ship to [country]?",
            "Is this product available?",
            "What are the ingredients?",
            "Do you carry this in [color]?",
            "How much does this cost?",
            "When will this be back in stock?",
            "Do you have the Samsung Galaxy S5?",
        ],
    ),
    "complaint_feedback": IntentSpec(
        id="complaint_feedback",
        name="Complaint / frustrated feedback",
        description="Angry, frustrated, or disappointed messages, often rants without a crisp ask.",
        escalate_default=True,  # escalates: needs human de-escalation
        seeds=[
            "Your customer service is terrible.",
            "I am very disappointed with Amazon.",
            "This is unacceptable. Fix it now.",
            "Worst experience ever.",
            "I've been waiting 6 months for an answer.",
            "This is the worst customer service.",
            "I'm very frustrated with this situation.",
            "You make customers fool in every sale.",
            "This isn't good enough.",
            "What worse service by amazon.",
            "Legit never buying anything again.",
            "I'm not paying for this.",
        ],
    ),
    "general_acknowledgement": IntentSpec(
        id="general_acknowledgement",
        name="Acknowledgement / small talk",
        description="Thanks, confirmations, small talk, off-topic, or empty acknowledgments.",
        escalate_default=False,
        seeds=[
            "Thank you!",
            "Thanks for the help.",
            "Ok thank you.",
            "My order was delivered. Thx.",
            "Great, thanks.",
            "You guys are amazing at customer service #grateful #thanks",
            "I really appreciate the help.",
        ],
    ),
}

INTENT_ORDER = list(INTENTS.keys())
ESCALATION_REASONS = {
    "account_access": "account/billing details cannot be shared over Twitter",
    "billing_payment": "payment details require a private, secure channel",
    "damage_defective": "claim needs order/photo evidence reviewed by a human",
    "complaint_feedback": "frustrated customer benefits from human de-escalation",
    "order_status": "order lookup requires private account details",
    "return_refund": "return/refund needs order confirmation and human handling",
}


class RedFlag:
    """Keywords that push a message toward escalation regardless of intent."""
    WORDS = [
        "fuck", "shit", "bitch", "pathetic", "worst", "terrible", "scam",
        "fraud", "urgent", "immediately", "lawyer", "legal", "sue", "lawsuit",
        "threat", "death threat", "sueing", "complaint", "refund",
    ]


def red_flags(text: str) -> list[str]:
    t = text.lower()
    found = []
    for w in RedFlag.WORDS:
        if w in t:
            found.append(w)
    return found
