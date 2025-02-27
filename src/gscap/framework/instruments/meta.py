from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    GRAINS = "grains"
    SOFTS = "soft"
    ENERGY = "energy"
    EQUITY = "equity"


class ContractType(Enum):
    FUTURES = "futures"
    OPTION = "option"


class Currency(Enum):
    USD = "usd"
    GBP = "gbp"


class Exchange(Enum):
    CME = "cme"
    CBOT = "cbot"


@dataclass
class InstrumentMetadata:
    symbol: str
    exchange: Exchange
    tick_value: float
    dem: float
    category: Category
    contract_type: ContractType
    quote_currency: Currency
    # fetch_symbol: str = None

    # def __post_init__(self):
    #     # If fetch_symbol is not provided, set it equal to symbol
    #     if self.fetch_symbol is None:
    #         self.fetch_symbol = self.symbol

    def __eq__(self, value):
        if (self.exchange, self.symbol) == (value.exchange, value.symbol):
            return True
        return False

    def __hash__(self):
        return hash(f"{self.exchange} {self.symbol}")


#############################################
#                                           #
#          DEFINING INSTRUMENTS             #
#                                           #
#############################################

mes = InstrumentMetadata(
    symbol="mes",
    # fetch_symbol="es",
    tick_value=1.250,
    dem=5,
    exchange=Exchange.CME,
    category=Category.EQUITY,
    contract_type=ContractType.FUTURES,
    quote_currency=Currency.USD,
)
es = InstrumentMetadata(
    symbol="es",
    tick_value=12.50,
    dem=50,
    exchange=Exchange.CME,
    category=Category.EQUITY,
    contract_type=ContractType.FUTURES,
    quote_currency=Currency.USD,
)

cl = InstrumentMetadata(
    symbol="cl",
    tick_value=10.00,
    dem=1000,
    exchange=Exchange.CME,
    category=Category.ENERGY,
    quote_currency=Currency.USD,
    contract_type=ContractType.FUTURES,
)


class AllInstruemntsMeta:
    MES = mes
    ES = es
    CL = cl

    @classmethod
    def list_instruments(cls) -> list:
        return [
            instrument
            for instrument in vars(cls).values()
            if isinstance(instrument, InstrumentMetadata)
        ]
