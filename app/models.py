from pydantic import BaseModel


class CreditCard(BaseModel):
    age: int
    depentent_count: str
    education: str
    income: str
    type_card: str
    book_month: int
    credit_relotionalship: int
    deactive_month_count: int
    contact_month_count: int
    credit_limit: int
    revolving_total: int
    avg_to_open_buy: int
    total_amt_4_1: int
    total_trans_amt: int
    total_trans_count: int
    total_count_4_1:int
    avg_utilization_ratio: int


class Phone(BaseModel):
    battery: int
    bluetooth: int
    microprossor: int
    dual_sum: int
    front_camera: int
    four_g: int
    memory: int
    depth: int
    weight: int
    pross_count: int
    main_camera_pk: int
    pk_height: int
    pk_en: int
    ram: int
    mobil_screen_height: int
    mobil_screeen_en: int
    battery_time: int
    three_g: int
    sensor: int
    wifi: int
