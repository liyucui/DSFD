from lib.helper.logger import logger
from lib.core.base_trainer.net_work import trainner
import setproctitle

# test

logger.info('train start')
setproctitle.setproctitle("detect")

trainner=trainner()

trainner.train()
