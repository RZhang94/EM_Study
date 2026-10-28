from signal import Position, Signal

def main():
    if __name__ == '__main__':
        tx = Signal("tx", frequency = 1e9, time0 = 0, timef = 0.1, position = Position(0, 0, 0))
        obj = tx.transmission(Position(0, 0, -0.10), "rx")
        rx = obj.transmission(tx.pos, "rx")

        print(rx.time0)


if __name__ == "__main__":
    main()
