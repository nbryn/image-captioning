import time
from src.config import get_config, MODE
from src.question_generation import QuestionGeneration
from src.generate_bert_data import generate_bert_data
from src.util import configure_logger, log

if __name__ == '__main__':
    args = get_config().parse_args()
    configure_logger(args.log_path)

    log(f"args: {args}")

    # time function
    start_time = time.time()

    if args.mode == MODE.ALL or args.mode == MODE.GEN_DATA:
        annotations = generate_bert_data(args)

        log("--- generate_bert_data took: %s seconds ---" %
            (time.time() - start_time))

    if args.mode == MODE.GEN_DATA:
        log("Exiting after generating data")
        exit()

    qg = QuestionGeneration(args)

    if args.mode == MODE.ALL or args.mode == MODE.FINE_TUNE:
        # -----Train qg-----
        start_time = time.time()
        qg.train()
        log("---train took: %s seconds ---" % (time.time() - start_time))

    if args.mode == MODE.TEST or args.mode == MODE.ALL or args.mode == MODE.FINE_TUNE:
        # -----Test qg-----
        start_time = time.time()
        qg.test()
        log("---test took: %s seconds ---" % (time.time() - start_time))
