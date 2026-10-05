import json
import numpy as np
import pandas as pd
import torch
import datasets
datasets.config.TORCHVISION_AVAILABLE = False
from transformers import AutoTokenizer, AutoModelForSequenceClassification

CLF_PATH = "/Users/fvcg/projects/uni-cognit-systems/train/banking77/distilroberta-banking77-saved"
TEST_CSV = "/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/test.csv"
OUT_CSV = "/Users/fvcg/projects/uni-cognit-systems/datasets/banking77/misclassified_samples.csv"

INTENT_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "pin_blocked",
    "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits", "top_up_reverted",
    "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity", "verify_source_of_funds",
    "verify_top_up", "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]
assert len(INTENT_NAMES) == 77, len(INTENT_NAMES)

clfTokenizer = AutoTokenizer.from_pretrained(CLF_PATH)
clfModel = AutoModelForSequenceClassification.from_pretrained(CLF_PATH)

test_df = pd.read_csv(TEST_CSV)

device = "mps" if torch.backends.mps.is_available() else "cpu"
clfModel.to(device)
clfModel.eval()


def tokenize_clf(texts):
    return clfTokenizer(texts, padding="max_length", truncation=True,
                        max_length=64, return_tensors="pt")


all_probs = []
all_preds = []
BS = 32
with torch.no_grad():
    for i in range(0, len(test_df), BS):
        texts = test_df["text"].iloc[i:i + BS].tolist()
        enc = tokenize_clf(texts)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = clfModel(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(-1).cpu().numpy())

probs = np.concatenate(all_probs, axis=0)
preds = np.concatenate(all_preds, axis=0)
true = test_df["label"].to_numpy()

df_pred = pd.DataFrame({
    "text": test_df["text"],
    "true_label_id": true,
    "pred_label_id": preds,
})
df_pred["true_label"] = df_pred["true_label_id"].map(lambda x: INTENT_NAMES[x])
df_pred["pred_label"] = df_pred["pred_label_id"].map(lambda x: INTENT_NAMES[x])
df_pred["confidence"] = probs[np.arange(len(probs)), preds]

mis = df_pred[df_pred["true_label_id"] != df_pred["pred_label_id"]].copy()
mis = mis.sort_values("confidence", ascending=False).head(20).reset_index(drop=True)
mis = mis[["text", "true_label", "pred_label", "confidence"]]
mis.to_csv(OUT_CSV, index=False)

print(f"Total test: {len(df_pred)}")
print(f"Mal clasificadas: {len(df_pred[df_pred['true_label_id'] != df_pred['pred_label_id']])}")
print(f"Guardadas 20 muestras en: {OUT_CSV}")
print()
print(mis.to_string())
