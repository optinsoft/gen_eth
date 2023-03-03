from eth_account import Account
# import secrets
# priv = secrets.token_hex(32)
# private_key = "0x" + priv
private_key = "0x726cc9f00e9a12eda1ad2bb1e778d40034da2e171082cf663d57b3b77c054da7"
print ("SAVE BUT DO NOT SHARE THIS:", private_key)
acct = Account.from_key(private_key)
print("Address:", acct.address) # 0x02bCb427D68353E91d31047102153fd086a74242